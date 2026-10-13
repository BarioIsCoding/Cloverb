// Cloverb standalone engine: the full network forward pass in C (no PyTorch), behind a UCI loop.
//
// Large GEMMs (qkv, attention output, MLP) run on Intel AMX bf16 tiles with weights packed once into
// VNNI tiles; attention (64x64 per head), norms, conditioning and the policy head run in fp32 AVX-512
// (the policy head stays fp32 so near-equal move scores do not tie). Falls back to fp32 AVX-512 GEMMs
// when AMX is unavailable. Input encoding and legal moves come from cloverb_core.c.
//
// Build: gcc -O3 -march=native -o cloverb csrc/cloverb_engine.c -lm
// Run:   ./cloverb weights.bin            (export with tools/export_engine.py)
#include "cloverb_core.c"

#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define NT 64
#define MAXL 32
#define EPS32 1.1920928955078125e-07f
#define EPS_QK 0.0078125f  // RMSNorm eps of bf16 q/k during training (torch.finfo(bfloat16).eps)

static int use_amx = 0;

// ------------------------------------------------------------------------------------------ weights
typedef struct { char name[128]; int ndim, dims[4]; float *data; } Tensor;
static Tensor *T; static int nT;
static int CFG[32];

static float *W(const char *name) {
    for (int i = 0; i < nT; i++) if (!strcmp(T[i].name, name)) return T[i].data;
    fprintf(stderr, "missing tensor %s\n", name); exit(1);
}
static float *Wf(const char *fmt, int l) { char b[128]; snprintf(b, sizeof b, fmt, l); return W(b); }

static void load_weights(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    char mg[4]; int ver, nc;
    if (fread(mg, 1, 4, f) != 4 || memcmp(mg, "CLVB", 4)) { fprintf(stderr, "bad weights file\n"); exit(1); }
    if (fread(&ver, 4, 1, f) != 1 || fread(&nc, 4, 1, f) != 1 || nc > 32 || fread(CFG, 4, nc, f) != (size_t)nc) exit(1);
    if (fread(&nT, 4, 1, f) != 1) exit(1);
    T = calloc(nT, sizeof(Tensor));
    for (int i = 0; i < nT; i++) {
        int nl; if (fread(&nl, 4, 1, f) != 1 || nl >= 128) exit(1);
        if (fread(T[i].name, 1, nl, f) != (size_t)nl) exit(1);
        T[i].name[nl] = 0;
        if (fread(&T[i].ndim, 4, 1, f) != 1 || fread(T[i].dims, 4, T[i].ndim, f) != (size_t)T[i].ndim) exit(1);
        size_t n = 1; for (int k = 0; k < T[i].ndim; k++) n *= T[i].dims[k];
        T[i].data = aligned_alloc(64, (n * 4 + 63) / 64 * 64);
        if (fread(T[i].data, 4, n, f) != n) exit(1);
    }
    fclose(f);
}

// ------------------------------------------------------------------------------------------ GEMM
// Linear y[64 x N] = x[64 x K] W^T (+ b), torch weight W is [N x K].
typedef struct { int N, K; uint16_t *vnni; float *wt; const float *b; } Lin;

static inline uint16_t f2bf(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    u += 0x7fff + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}

static Lin make_lin_ex(const float *w, const float *b, int N, int K, int fp32) {
    Lin L = {N, K, NULL, NULL, b};
    if (!fp32 && use_amx && N % 16 == 0 && K % 32 == 0) {
        // tile (nb, kb): 16 rows r (k pairs) x 16 cols n x 2 -> 512 bf16, stored [nb][kb]
        L.vnni = aligned_alloc(64, (size_t)N * K * 2);
        int KB = K / 32;
        for (int nb = 0; nb < N / 16; nb++)
            for (int kb = 0; kb < KB; kb++) {
                uint16_t *t = L.vnni + ((size_t)nb * KB + kb) * 512;
                for (int r = 0; r < 16; r++)
                    for (int n = 0; n < 16; n++)
                        for (int j = 0; j < 2; j++)
                            t[r * 32 + n * 2 + j] = f2bf(w[(size_t)(nb * 16 + n) * K + kb * 32 + 2 * r + j]);
            }
    } else {
        L.wt = aligned_alloc(64, (size_t)N * K * 4);            // K x N, for the AVX-512 path
        for (int n = 0; n < N; n++) for (int k = 0; k < K; k++) L.wt[(size_t)k * N + n] = w[(size_t)n * K + k];
    }
    return L;
}
static Lin make_lin(const float *w, const float *b, int N, int K) { return make_lin_ex(w, b, N, K, 0); }

typedef struct { uint8_t palette, start_row, res[14]; uint16_t colsb[16]; uint8_t rows[16]; } __attribute__((packed)) TileCfg;

static void amx_config(void) {
    TileCfg c; memset(&c, 0, sizeof c);
    c.palette = 1;
    for (int i = 0; i < 8; i++) { c.rows[i] = 16; c.colsb[i] = 64; }
    _tile_loadconfig(&c);
}

static uint16_t XB[NT * 4096] __attribute__((aligned(64)));

static void linear(float *y, const float *x, const Lin *L) {
    int N = L->N, K = L->K;
    if (L->vnni) {
        for (int i = 0; i < NT * K; i += 16)
            _mm256_store_si256((__m256i *)(XB + i), (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(x + i)));
        int KB = K / 32;
        for (int mb = 0; mb < NT; mb += 32)
            for (int nb = 0; nb < N / 16; nb += 2) {
                _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
                for (int kb = 0; kb < KB; kb++) {
                    _tile_loadd(4, XB + (size_t)mb * K + kb * 32, K * 2);
                    _tile_loadd(5, XB + (size_t)(mb + 16) * K + kb * 32, K * 2);
                    _tile_loadd(6, L->vnni + ((size_t)nb * KB + kb) * 512, 64);
                    _tile_loadd(7, L->vnni + ((size_t)(nb + 1) * KB + kb) * 512, 64);
                    _tile_dpbf16ps(0, 4, 6); _tile_dpbf16ps(1, 4, 7);
                    _tile_dpbf16ps(2, 5, 6); _tile_dpbf16ps(3, 5, 7);
                }
                _tile_stored(0, y + (size_t)mb * N + nb * 16, N * 4);
                _tile_stored(1, y + (size_t)mb * N + nb * 16 + 16, N * 4);
                _tile_stored(2, y + (size_t)(mb + 16) * N + nb * 16, N * 4);
                _tile_stored(3, y + (size_t)(mb + 16) * N + nb * 16 + 16, N * 4);
            }
    } else {
        for (int i = 0; i < NT; i += 4)
            for (int n = 0; n < N; n += 16) {
                __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
                for (int k = 0; k < K; k++) {
                    __m512 w = _mm512_loadu_ps(L->wt + (size_t)k * N + n);
                    a0 = _mm512_fmadd_ps(_mm512_set1_ps(x[(size_t)i * K + k]), w, a0);
                    a1 = _mm512_fmadd_ps(_mm512_set1_ps(x[(size_t)(i + 1) * K + k]), w, a1);
                    a2 = _mm512_fmadd_ps(_mm512_set1_ps(x[(size_t)(i + 2) * K + k]), w, a2);
                    a3 = _mm512_fmadd_ps(_mm512_set1_ps(x[(size_t)(i + 3) * K + k]), w, a3);
                }
                _mm512_storeu_ps(y + (size_t)i * N + n, a0); _mm512_storeu_ps(y + (size_t)(i + 1) * N + n, a1);
                _mm512_storeu_ps(y + (size_t)(i + 2) * N + n, a2); _mm512_storeu_ps(y + (size_t)(i + 3) * N + n, a3);
            }
    }
    if (L->b)
        for (int i = 0; i < NT; i++) for (int n = 0; n < N; n++) y[(size_t)i * N + n] += L->b[n];
}

// fp32 matrix-vector y[N] = W[N x K] x[K] + b
static void matvec(float *y, const float *w, const float *b, const float *x, int N, int K) {
    for (int n = 0; n < N; n++) {
        __m512 acc = _mm512_setzero_ps();
        int k = 0;
        for (; k + 16 <= K; k += 16) acc = _mm512_fmadd_ps(_mm512_loadu_ps(w + (size_t)n * K + k), _mm512_loadu_ps(x + k), acc);
        float s = _mm512_reduce_add_ps(acc);
        for (; k < K; k++) s += w[(size_t)n * K + k] * x[k];
        y[n] = s + (b ? b[n] : 0.f);
    }
}

static inline float silu(float v) { return v / (1.f + expf(-v)); }

static void rmsnorm_rows(float *x, const float *w, int rows, int d, float eps) {
    for (int i = 0; i < rows; i++) {
        float *r = x + (size_t)i * d, ss = 0;
        for (int k = 0; k < d; k++) ss += r[k] * r[k];
        float inv = 1.f / sqrtf(ss / d + eps);
        for (int k = 0; k < d; k++) r[k] = r[k] * inv * (w ? w[k] : 1.f);
    }
}

// ------------------------------------------------------------------------------------------ model
typedef struct {
    float *stat, *n1, *n2, *qn, *kn, *adw, *adb;
    Lin qkv, proj, fc1, fc2;
} Layer;

static int D, NL, NH, HID, HIST, DC, NCOND, VH, TB, POSTLN, CAPTYPE, ADALN;
static Layer LY[MAXL];
static Lin LPF, LPT;
static float *E_sq, *E_castle, *E_ep, *E_piece, *E_hfrom, *E_hto, *E_hpiece, *E_hcap;
static float *C0w, *C0b, *C2w, *C2b, *CTw, *CTb, *INNORM, *FNORM, *PF, *PT, *PP, *V0w, *V0b, *V2w, *V2b, *T0w, *T0b, *T2w, *T2b;

static void build_model(void) {
    D = CFG[0]; NL = CFG[1]; NH = CFG[2]; HID = CFG[3]; HIST = CFG[4]; DC = CFG[5];
    if (CFG[6] != 0) { fprintf(stderr, "dynamic templates are not supported by the C engine\n"); exit(1); }
    ADALN = CFG[8]; POSTLN = CFG[9]; CAPTYPE = CFG[10]; VH = CFG[11]; TB = CFG[12]; NCOND = CFG[14];
    if (!CFG[13] || !ADALN || D % 32 || NL > MAXL) { fprintf(stderr, "unsupported configuration\n"); exit(1); }
    E_sq = W("square"); E_castle = W("castle"); E_ep = W("ep"); E_piece = W("piece.weight");
    E_hfrom = W("h_from.weight"); E_hto = W("h_to.weight"); E_hpiece = W("h_piece.weight"); E_hcap = W("h_cap.weight");
    C0w = W("cond.0.weight"); C0b = W("cond.0.bias"); C2w = W("cond.2.weight"); C2b = W("cond.2.bias");
    CTw = W("cond_tok.weight"); CTb = W("cond_tok.bias");
    INNORM = POSTLN ? W("in_norm.weight") : NULL; FNORM = W("norm.weight");
    PF = W("p_from.weight"); PT = W("p_to.weight"); PP = W("p_promo.weight");
    V0w = W("v_head.0.weight"); V0b = W("v_head.0.bias"); V2w = W("v_head.2.weight"); V2b = W("v_head.2.bias");
    T0w = W("t_head.0.weight"); T0b = W("t_head.0.bias"); T2w = W("t_head.2.weight"); T2b = W("t_head.2.bias");
    for (int l = 0; l < NL; l++) {
        Layer *y = &LY[l];
        y->stat = Wf("blocks.%d.static", l);
        y->n1 = POSTLN ? Wf("blocks.%d.norm1.weight", l) : NULL;
        y->n2 = POSTLN ? Wf("blocks.%d.norm2.weight", l) : NULL;
        y->qn = Wf("blocks.%d.q_norm.weight", l); y->kn = Wf("blocks.%d.k_norm.weight", l);
        y->adw = Wf("blocks.%d.adaln.weight", l); y->adb = Wf("blocks.%d.adaln.bias", l);
        y->qkv = make_lin(Wf("blocks.%d.qkv.weight", l), NULL, 3 * D, D);
        y->proj = make_lin(Wf("blocks.%d.proj.weight", l), NULL, D, D);
        y->fc1 = make_lin(Wf("blocks.%d.fc1.weight", l), NULL, HID, D);
        y->fc2 = make_lin(Wf("blocks.%d.fc2.weight", l), NULL, D, HID);
    }
    LPF = make_lin_ex(PF, NULL, D, D, 1); LPT = make_lin_ex(PT, NULL, D, D, 1);
}

static const float ELO_A[15] = {400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200};
static const float TSEC[14] = {2, 5, 10, 20, 30, 45, 60, 90, 120, 180, 300, 600, 1200, 3600};

static void hat(float *out, float x, const float *a, int n) {
    if (x < a[0]) x = a[0];
    if (x > a[n - 1]) x = a[n - 1];
    for (int i = 0; i < n; i++) {
        float v;
        if (x <= a[i]) v = i == 0 ? 1.f : 1.f - (a[i] - x) / (a[i] - a[i - 1]);
        else v = i == n - 1 ? 1.f : 1.f - (x - a[i]) / (a[i + 1] - a[i]);
        out[i] = v < 0 ? 0 : (v > 1 ? 1 : v);
    }
}

typedef struct { float logits[4352]; float wdl[3]; float tdist[64]; } Output;

static float X[NT * 1024], Hh[NT * 1024], QKV[NT * 3072], O[NT * 1024], A[NT * 1024], MLP[NT * 4096];

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

// attention for one layer: QKV [64 x 3D] -> O [64 x D]; head dim must be a multiple of 16 (<= 64)
static void attention(const Layer *y) {
    int dh = D / NH, nv = dh / 16;
    float scale = 1.f / sqrtf((float)dh);
    static float q[NT * 64] __attribute__((aligned(64))), kt[64 * NT] __attribute__((aligned(64)));
    static float v[NT * 64] __attribute__((aligned(64))), s[NT * NT] __attribute__((aligned(64)));
    for (int h = 0; h < NH; h++) {
        for (int i = 0; i < NT; i++) {
            const float *r = QKV + (size_t)i * 3 * D + h * dh;
            __m512 sq = _mm512_setzero_ps(), sk = sq;
            for (int c = 0; c < dh; c += 16) {
                __m512 a = _mm512_loadu_ps(r + c), b = _mm512_loadu_ps(r + D + c);
                sq = _mm512_fmadd_ps(a, a, sq); sk = _mm512_fmadd_ps(b, b, sk);
            }
            float iq = 1.f / sqrtf(_mm512_reduce_add_ps(sq) / dh + EPS_QK), ik = 1.f / sqrtf(_mm512_reduce_add_ps(sk) / dh + EPS_QK);
            for (int c = 0; c < dh; c++) {
                q[i * dh + c] = r[c] * iq * y->qn[c] * scale;
                kt[c * NT + i] = r[D + c] * ik * y->kn[c];
                v[i * dh + c] = r[2 * D + c];
            }
        }
        const float *st = y->stat + (size_t)h * NT * NT;
        for (int i = 0; i < NT; i += 2) {                      // scores + softmax, 2 rows at a time
            __m512 a[2][4];
            for (int r = 0; r < 2; r++) for (int c = 0; c < 4; c++) a[r][c] = _mm512_loadu_ps(st + (i + r) * NT + 16 * c);
            for (int c = 0; c < dh; c++) {
                __m512 q0 = _mm512_set1_ps(q[i * dh + c]), q1 = _mm512_set1_ps(q[(i + 1) * dh + c]);
                for (int u = 0; u < 4; u++) {
                    __m512 kk = _mm512_load_ps(kt + c * NT + 16 * u);
                    a[0][u] = _mm512_fmadd_ps(q0, kk, a[0][u]); a[1][u] = _mm512_fmadd_ps(q1, kk, a[1][u]);
                }
            }
            for (int r = 0; r < 2; r++) {
                __m512 m = _mm512_set1_ps(_mm512_reduce_max_ps(_mm512_max_ps(_mm512_max_ps(a[r][0], a[r][1]), _mm512_max_ps(a[r][2], a[r][3]))));
                __m512 sum = _mm512_setzero_ps();
                for (int u = 0; u < 4; u++) { a[r][u] = exp512(_mm512_sub_ps(a[r][u], m)); sum = _mm512_add_ps(sum, a[r][u]); }
                __m512 inv = _mm512_set1_ps(1.f / _mm512_reduce_add_ps(sum));
                for (int u = 0; u < 4; u++) _mm512_store_ps(s + (i + r) * NT + 16 * u, _mm512_mul_ps(a[r][u], inv));
            }
        }
        for (int i = 0; i < NT; i += 4) {                      // O = P V, 4 rows at a time
            __m512 acc[4][4];
            for (int r = 0; r < 4; r++) for (int c = 0; c < nv; c++) acc[r][c] = _mm512_setzero_ps();
            for (int j = 0; j < NT; j++) {
                __m512 p0 = _mm512_set1_ps(s[i * NT + j]), p1 = _mm512_set1_ps(s[(i + 1) * NT + j]);
                __m512 p2 = _mm512_set1_ps(s[(i + 2) * NT + j]), p3 = _mm512_set1_ps(s[(i + 3) * NT + j]);
                for (int c = 0; c < nv; c++) {
                    __m512 vv = _mm512_load_ps(v + j * dh + 16 * c);
                    acc[0][c] = _mm512_fmadd_ps(p0, vv, acc[0][c]); acc[1][c] = _mm512_fmadd_ps(p1, vv, acc[1][c]);
                    acc[2][c] = _mm512_fmadd_ps(p2, vv, acc[2][c]); acc[3][c] = _mm512_fmadd_ps(p3, vv, acc[3][c]);
                }
            }
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < nv; c++) _mm512_storeu_ps(O + (size_t)(i + r) * D + h * dh + 16 * c, acc[r][c]);
        }
    }
}

static void modulate(float *out, const float *x, const float *sc, const float *sh) {
    for (int i = 0; i < NT; i++)
        for (int c = 0; c < D; c += 16) {
            __m512 v = _mm512_loadu_ps(x + i * D + c);
            _mm512_storeu_ps(out + i * D + c, _mm512_fmadd_ps(v, _mm512_add_ps(_mm512_set1_ps(1.f), _mm512_loadu_ps(sc + c)), _mm512_loadu_ps(sh + c)));
        }
}

static void gated_add(float *x, const float *a, const float *g) {
    for (int i = 0; i < NT; i++)
        for (int c = 0; c < D; c += 16) {
            __m512 gg = _mm512_add_ps(_mm512_set1_ps(1.f), _mm512_loadu_ps(g + c));
            _mm512_storeu_ps(x + i * D + c, _mm512_fmadd_ps(_mm512_loadu_ps(a + i * D + c), gg, _mm512_loadu_ps(x + i * D + c)));
        }
}

// Forward pass for one encoded position (board/meta/hist as produced by cb_encode_position).
static void forward(const int8_t *board, const int16_t *meta, const int8_t *hist, float elo, float base, float inc,
                    float clock, Output *out) {
    // ---- embedding
    for (int s = 0; s < NT; s++)
        for (int c = 0; c < D; c++) X[s * D + c] = E_piece[board[s] * D + c] + E_sq[s * D + c];
    static const int csq[4] = {7, 0, 63, 56};
    for (int r = 0; r < 4; r++) if (meta[r]) for (int c = 0; c < D; c++) X[csq[r] * D + c] += E_castle[r * D + c];
    if (meta[4] >= 0) for (int c = 0; c < D; c++) X[meta[4] * D + c] += E_ep[c];
    int hlen = 0;
    for (int j = 0; j < HIST; j++) {
        const int8_t *h = hist + j * 4;
        if (h[0] < 0) continue;
        hlen++;
        int side = (j % 2 == 0), pt = h[2], cap = h[3] > 0 ? h[3] : 0;
        const float *ef = E_hfrom + j * D, *et = E_hto + j * D;
        const float *pf = E_hpiece + (side * 6 + pt) * D, *ptv = E_hpiece + (12 + side * 6 + pt) * D;
        for (int c = 0; c < D; c++) { X[h[0] * D + c] += ef[c] + pf[c]; X[h[1] * D + c] += et[c] + ptv[c]; }
        if (cap) {
            const float *ec = E_hcap + (CAPTYPE ? side * 6 + cap : side) * D;
            for (int c = 0; c < D; c++) X[h[1] * D + c] += ec[c];
        }
    }
    // ---- conditioning
    float f[64], c1[256], cvec[256];
    float lt = log1pf(clock);
    hat(f, elo, ELO_A, 15);
    float la[14]; for (int i = 0; i < 14; i++) la[i] = log1pf(TSEC[i]);
    hat(f + 15, lt, la, 14);
    float hm = meta[5] > 100 ? 100 : meta[5], fm = meta[6] > 150 ? 150 : meta[6], rel = clock / (base + 40 * inc + 1);
    f[29] = log1pf(base) / 8; f[30] = log1pf(inc) / 4; f[31] = lt / 8;
    f[32] = rel < 0 ? 0 : (rel > 2 ? 2 : rel); f[33] = hm / 100; f[34] = fm / 100;
    for (int r = 0; r < 4; r++) f[35 + r] = meta[r];
    f[39] = meta[7]; f[40] = hlen / 64.f;
    matvec(c1, C0w, C0b, f, DC, NCOND); for (int i = 0; i < DC; i++) c1[i] = silu(c1[i]);
    matvec(cvec, C2w, C2b, c1, DC, DC); for (int i = 0; i < DC; i++) cvec[i] = silu(cvec[i]);
    float ct[1024]; matvec(ct, CTw, CTb, cvec, D, DC);
    for (int s = 0; s < NT; s++) for (int c = 0; c < D; c++) X[s * D + c] += ct[c];
    if (POSTLN) rmsnorm_rows(X, INNORM, NT, D, EPS32);
    // ---- blocks (post-norm, adaLN-Zero modulation)
    static float ada[6 * 1024];
    for (int l = 0; l < NL; l++) {
        Layer *y = &LY[l];
        matvec(ada, y->adw, y->adb, cvec, 6 * D, DC);
        const float *s1 = ada, *b1 = ada + D, *g1 = ada + 2 * D, *s2 = ada + 3 * D, *b2 = ada + 4 * D, *g2 = ada + 5 * D;
        if (POSTLN) {
            modulate(Hh, X, s1, b1);
        } else {
            memcpy(Hh, X, sizeof(float) * NT * D); rmsnorm_rows(Hh, NULL, NT, D, EPS32);
            modulate(Hh, Hh, s1, b1);
        }
        linear(QKV, Hh, &y->qkv);
        attention(y);
        linear(A, O, &y->proj);
        gated_add(X, A, g1);
        if (POSTLN) {
            rmsnorm_rows(X, y->n1, NT, D, EPS32);
            modulate(Hh, X, s2, b2);
        } else {
            memcpy(Hh, X, sizeof(float) * NT * D); rmsnorm_rows(Hh, NULL, NT, D, EPS32);
            modulate(Hh, Hh, s2, b2);
        }
        linear(MLP, Hh, &y->fc1);
        for (int i = 0; i < NT * HID; i++) { float v = MLP[i] > 0 ? MLP[i] : 0; MLP[i] = v * v; }
        linear(A, MLP, &y->fc2);
        gated_add(X, A, g2);
        if (POSTLN) rmsnorm_rows(X, y->n2, NT, D, EPS32);
    }
    rmsnorm_rows(X, FNORM, NT, D, EPS32);
    // ---- heads (fp32)
    static float FF[NT * 1024], TT[NT * 1024];
    linear(FF, X, &LPF); linear(TT, X, &LPT);
    float isq = 1.f / sqrtf((float)D);
    for (int i = 0; i < NT; i++)
        for (int j = 0; j < NT; j++) {
            __m512 acc = _mm512_setzero_ps();
            for (int c = 0; c < D; c += 16) acc = _mm512_fmadd_ps(_mm512_loadu_ps(FF + i * D + c), _mm512_loadu_ps(TT + j * D + c), acc);
            out->logits[i * 64 + j] = _mm512_reduce_add_ps(acc) * isq;
        }
    for (int tf = 0; tf < 8; tf++) {
        float pr[4]; matvec(pr, PP, NULL, TT + (56 + tf) * D, 4, D);
        for (int ff = 0; ff < 8; ff++)
            for (int p = 0; p < 4; p++) out->logits[4096 + (ff * 8 + tf) * 4 + p] = out->logits[(48 + ff) * 64 + 56 + tf] + pr[p];
    }
    float g[2048], hv[1024], v3[64];
    for (int c = 0; c < D; c++) { float s = 0; for (int i = 0; i < NT; i++) s += X[i * D + c]; g[c] = s / NT; }
    for (int c = 0; c < DC; c++) g[D + c] = cvec[c];
    matvec(hv, V0w, V0b, g, VH, D + DC); for (int i = 0; i < VH; i++) hv[i] = silu(hv[i]);
    matvec(v3, V2w, V2b, hv, 3, VH);
    float m = fmaxf(v3[0], fmaxf(v3[1], v3[2])), z = 0;
    for (int i = 0; i < 3; i++) { out->wdl[i] = expf(v3[i] - m); z += out->wdl[i]; }
    for (int i = 0; i < 3; i++) out->wdl[i] /= z;                      // loss, draw, win (mover)
    matvec(hv, T0w, T0b, g, VH, D + DC); for (int i = 0; i < VH; i++) hv[i] = silu(hv[i]);
    matvec(v3, T2w, T2b, hv, TB, VH);
    m = -1e30f; for (int i = 0; i < TB; i++) m = fmaxf(m, v3[i]);
    z = 0; for (int i = 0; i < TB; i++) { out->tdist[i] = expf(v3[i] - m); z += out->tdist[i]; }
    for (int i = 0; i < TB; i++) out->tdist[i] /= z;
}

// ------------------------------------------------------------------------------------------ engine
static void move_str(uint16_t mv, char *s) {
    int f = mv & 63, t = (mv >> 6) & 63, p = mv >> 12;
    s[0] = 'a' + (f & 7); s[1] = '1' + (f >> 3); s[2] = 'a' + (t & 7); s[3] = '1' + (t >> 3);
    s[4] = p ? " qrbn"[p] : 0; s[5] = 0;
}

static int parse_move(const char *u, uint16_t *out) {
    if (strlen(u) < 4) return 0;
    int f = (u[0] - 'a') + 8 * (u[1] - '1'), t = (u[2] - 'a') + 8 * (u[3] - '1'), p = 0;
    if (f < 0 || f > 63 || t < 0 || t > 63) return 0;
    if (u[4]) { const char *pc = strchr(" qrbn", u[4]); p = pc ? (int)(pc - " qrbn") : 0; }
    *out = (uint16_t)(f | (t << 6) | (p << 12));
    return 1;
}

typedef struct { uint16_t mv; float p; } Cand;

static int cmp_cand(const void *a, const void *b) {
    float d = ((const Cand *)b)->p - ((const Cand *)a)->p;
    return d > 0 ? 1 : d < 0 ? -1 : 0;
}

// Evaluate a position; fills cands sorted by probability. Returns number of legal moves (<0 on error).
static Output OUT;
static int evaluate(const char *fen, const uint16_t *mv, int n, float elo, float base, float inc, float clock, Cand *cands,
                    double *ms) {
    int8_t board[64], hist[64 * 4 * 4]; int16_t meta[8]; uint8_t legal[544]; uint16_t lm[256];
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    int nl = cb_encode_position(fen, mv, n, HIST, board, meta, hist, legal, lm);
    if (nl <= 0) return nl;
    int black = (fen && fen[0] ? (strchr(fen, ' ') && strchr(fen, ' ')[1] == 'b') : 0) ^ (n & 1);
    forward(board, meta, hist, elo, base, inc, clock, &OUT);
    float mx = -1e30f;
    for (int i = 0; i < nl; i++) { cands[i].mv = lm[i]; cands[i].p = OUT.logits[cb_vocab_index(lm[i], black)]; if (cands[i].p > mx) mx = cands[i].p; }
    float z = 0;
    for (int i = 0; i < nl; i++) { cands[i].p = expf(cands[i].p - mx); z += cands[i].p; }
    for (int i = 0; i < nl; i++) cands[i].p /= z;
    qsort(cands, nl, sizeof(Cand), cmp_cand);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (ms) *ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    return nl;
}

static float urand(void) { return (rand() + 0.5f) / ((float)RAND_MAX + 1.f); }

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s weights.bin [bench | dump FEN|startpos moves...]\n", argv[0]); return 1; }
    // AMX needs permission for the tile data state (Linux >= 5.16)
    if (__builtin_cpu_supports("amx-bf16") && syscall(SYS_arch_prctl, 0x1023, 18) == 0) use_amx = 1;
    load_weights(argv[1]);
    build_model();
    if (use_amx) amx_config();
    srand((unsigned)time(NULL));
    static Cand cands[256];
    if (argc > 2 && !strcmp(argv[2], "bench")) {  // latency benchmark on a middlegame position
        const char *mvs[] = {"e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7"};
        uint16_t mv[10]; for (int i = 0; i < 10; i++) parse_move(mvs[i], mv + i);
        double ms, tot = 0; int n = 200;
        for (int i = 0; i < 20; i++) evaluate(NULL, mv, 10, 1500, 180, 2, 150, cands, &ms);
        for (int i = 0; i < n; i++) { evaluate(NULL, mv, 10, 1500, 180, 2, 150, cands, &ms); tot += ms; }
        char s[8]; move_str(cands[0].mv, s);
        printf("%s: %.3f ms/position (1 thread), top move %s %.3f\n", use_amx ? "AMX bf16" : "AVX-512 fp32", tot / n, s, cands[0].p);
        return 0;
    }
    if (argc > 2 && !strcmp(argv[2], "dump")) {    // dump raw logits for testing: dump ELO BASE INC CLOCK moves...
        float elo = atof(argv[3]), base = atof(argv[4]), inc = atof(argv[5]), clk = atof(argv[6]);
        uint16_t mv[512]; int n = 0;
        for (int i = 7; i < argc && n < 512; i++) parse_move(argv[i], mv + n++);
        int8_t board[64], hist[64 * 4 * 4]; int16_t meta[8]; uint8_t legal[544]; uint16_t lm[256];
        if (cb_encode_position(NULL, mv, n, HIST, board, meta, hist, legal, lm) < 0) return 1;
        forward(board, meta, hist, elo, base, inc, clk, &OUT);
        fwrite(OUT.logits, 4, 4352, stdout); fwrite(OUT.wdl, 4, 3, stdout); fwrite(OUT.tdist, 4, TB, stdout);
        return 0;
    }
    // ---- UCI
    char line[65536], fen[256] = "";
    static uint16_t moves[2048]; int nmoves = 0;
    float elo = 1500, temp = 0, base_opt = 0, game_base = -1;
    int multipv = 1;
    setvbuf(stdout, NULL, _IONBF, 0);
    while (fgets(line, sizeof line, stdin)) {
        char *nl = strchr(line, '\n'); if (nl) *nl = 0;
        if (!strncmp(line, "uci", 3) && (line[3] == 0 || line[3] == ' ')) {
            printf("id name Cloverb\nid author Cloverb authors\n");
            printf("option name Elo type spin default 1500 min 400 max 3200\n");
            printf("option name BaseTime type spin default 0 min 0 max 10800\n");
            printf("option name Temperature type string default 0\n");
            printf("option name MultiPV type spin default 1 min 1 max 20\nuciok\n");
        } else if (!strcmp(line, "isready")) printf("readyok\n");
        else if (!strcmp(line, "ucinewgame")) { nmoves = 0; fen[0] = 0; game_base = -1; }
        else if (!strncmp(line, "setoption", 9)) {
            char *nm = strstr(line, "name "), *vl = strstr(line, " value ");
            if (nm && vl) {
                *vl = 0; nm += 5; vl += 7;
                if (!strcasecmp(nm, "Elo")) elo = atof(vl);
                else if (!strcasecmp(nm, "Temperature")) temp = atof(vl);
                else if (!strcasecmp(nm, "MultiPV")) multipv = atoi(vl);
                else if (!strcasecmp(nm, "BaseTime")) base_opt = atof(vl);
            }
        } else if (!strncmp(line, "position", 8)) {
            nmoves = 0; fen[0] = 0;
            char *p = strstr(line, "fen "), *m = strstr(line, " moves ");
            if (p) { p += 4; size_t L2 = m ? (size_t)(m - p) : strlen(p); if (L2 > 255) L2 = 255; memcpy(fen, p, L2); fen[L2] = 0; }
            if (m) { char *tok = strtok(m + 7, " "); while (tok && nmoves < 2048) { if (parse_move(tok, moves + nmoves)) nmoves++; tok = strtok(NULL, " "); } }
        } else if (!strncmp(line, "go", 2)) {
            int black = (fen[0] ? (strchr(fen, ' ') && strchr(fen, ' ')[1] == 'b') : 0) ^ (nmoves & 1);
            float wt = 180000, bt = 180000, wi = 0, bi = 0;
            char *q;
            if ((q = strstr(line, "wtime "))) wt = atof(q + 6);
            if ((q = strstr(line, "btime "))) bt = atof(q + 6);
            if ((q = strstr(line, "winc "))) wi = atof(q + 5);
            if ((q = strstr(line, "binc "))) bi = atof(q + 5);
            float clk = (black ? bt : wt) / 1000, inc = (black ? bi : wi) / 1000;
            if (game_base < 0) game_base = fmaxf(wt, bt) / 1000;
            float base = base_opt > 0 ? base_opt : game_base;
            double ms;
            int n = evaluate(fen[0] ? fen : NULL, moves, nmoves, elo, base, inc, clk, cands, &ms);
            if (n <= 0) { printf("bestmove 0000\n"); continue; }
            int cp = (int)lrintf(400 * (OUT.wdl[2] - OUT.wdl[0]));
            for (int k = 0; k < multipv && k < n; k++) {
                char s[8]; move_str(cands[k].mv, s);
                printf("info depth 1 multipv %d score cp %d wdl %d %d %d nodes 1 time %d pv %s string policy %.3f\n", k + 1, cp,
                       (int)lrintf(1000 * OUT.wdl[2]), (int)lrintf(1000 * OUT.wdl[1]), (int)lrintf(1000 * OUT.wdl[0]), (int)ms, s, cands[k].p);
            }
            int pick = 0;
            if (temp > 0) {
                float z = 0, w[256];
                for (int k = 0; k < n; k++) { w[k] = powf(cands[k].p, 1.f / temp); z += w[k]; }
                float r = urand() * z;
                for (pick = 0; pick < n - 1 && (r -= w[pick]) > 0; pick++) {}
            }
            char s[8]; move_str(cands[pick].mv, s);
            printf("bestmove %s\n", s);
        } else if (!strcmp(line, "quit")) break;
    }
    return 0;
}
