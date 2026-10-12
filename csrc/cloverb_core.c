// Cloverb core: chess rules, PGN movetext parsing and training-batch construction in C.
//
// Everything Python needs from chess rules goes through three entry points (ctypes):
//   cb_parse_games   SAN movetext (Lichess style, with [%clk] comments) -> packed moves + clocks
//   cb_build_batch   (game, ply) samples -> mover-perspective board, 64-ply move history, legal mask
//   cb_build_maia    (game, ply) samples -> Maia-3 style 12-plane history bitboards (teacher inputs)
// Moves are packed as from | to << 6 | promo << 12 with absolute squares (a1=0, h8=63) and
// promo 0=none 1=q 2=r 3=b 4=n, the format of cloverb/gamestore.py.
// Build: gcc -O3 -march=native -shared -fPIC -o libcloverb.so cloverb_core.c
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

enum { EMPTY = 0, PAWN = 1, KNIGHT, BISHOP, ROOK, QUEEN, KING };
#define COLOR(p) ((p) >> 3)
#define TYPE(p) ((p) & 7)
#define MK(c, t) (((c) << 3) | (t))
#define FILE_OF(s) ((s) & 7)
#define RANK_OF(s) ((s) >> 3)

typedef struct {
    int8_t sq[64];
    int8_t stm;      // 0 white, 1 black
    int8_t castle;   // 1 wK, 2 wQ, 4 bK, 8 bQ
    int8_t ep;       // en passant target square or -1
    int8_t ksq[2];
    int16_t halfmove;
    int16_t fullmove;
} Board;

typedef struct { int8_t from, to, promo; } Move;  // promo: 0 none, else piece type (QUEEN..KNIGHT)

static int8_t N_TGT[64][8], N_CNT[64], K_TGT[64][8], K_CNT[64];
static int8_t RAY[64][8][7], RAY_LEN[64][8];
static const int DF[8] = {1, -1, 0, 0, 1, 1, -1, -1}, DR[8] = {0, 0, 1, -1, 1, -1, 1, -1};
static int inited = 0;

static void init_tables(void) {
    if (inited) return;
    static const int nf[8] = {1, 2, 2, 1, -1, -2, -2, -1}, nr[8] = {2, 1, -1, -2, -2, -1, 1, 2};
    for (int s = 0; s < 64; s++) {
        int f = FILE_OF(s), r = RANK_OF(s);
        N_CNT[s] = K_CNT[s] = 0;
        for (int i = 0; i < 8; i++) {
            int a = f + nf[i], b = r + nr[i];
            if (a >= 0 && a < 8 && b >= 0 && b < 8) N_TGT[s][N_CNT[s]++] = b * 8 + a;
            a = f + DF[i]; b = r + DR[i];
            if (a >= 0 && a < 8 && b >= 0 && b < 8) K_TGT[s][K_CNT[s]++] = b * 8 + a;
        }
        for (int d = 0; d < 8; d++) {
            int n = 0, a = f + DF[d], b = r + DR[d];
            while (a >= 0 && a < 8 && b >= 0 && b < 8) { RAY[s][d][n++] = b * 8 + a; a += DF[d]; b += DR[d]; }
            RAY_LEN[s][d] = n;
        }
    }
    inited = 1;
}

static void board_init(Board *B) {
    static const int8_t back[8] = {ROOK, KNIGHT, BISHOP, QUEEN, KING, BISHOP, KNIGHT, ROOK};
    memset(B, 0, sizeof(*B));
    for (int f = 0; f < 8; f++) {
        B->sq[f] = MK(0, back[f]); B->sq[8 + f] = MK(0, PAWN);
        B->sq[48 + f] = MK(1, PAWN); B->sq[56 + f] = MK(1, back[f]);
    }
    B->stm = 0; B->castle = 15; B->ep = -1; B->ksq[0] = 4; B->ksq[1] = 60; B->halfmove = 0; B->fullmove = 1;
}

// Is square s attacked by color c?
static int attacked(const Board *B, int s, int c) {
    int f = FILE_OF(s);
    if (c == 0) {
        if (f < 7 && s - 7 >= 0 && B->sq[s - 7] == MK(0, PAWN)) return 1;
        if (f > 0 && s - 9 >= 0 && B->sq[s - 9] == MK(0, PAWN)) return 1;
    } else {
        if (f > 0 && s + 7 < 64 && B->sq[s + 7] == MK(1, PAWN)) return 1;
        if (f < 7 && s + 9 < 64 && B->sq[s + 9] == MK(1, PAWN)) return 1;
    }
    for (int i = 0; i < N_CNT[s]; i++) if (B->sq[N_TGT[s][i]] == MK(c, KNIGHT)) return 1;
    for (int i = 0; i < K_CNT[s]; i++) if (B->sq[K_TGT[s][i]] == MK(c, KING)) return 1;
    for (int d = 0; d < 8; d++) {
        int slider = d < 4 ? ROOK : BISHOP;
        for (int i = 0; i < RAY_LEN[s][d]; i++) {
            int p = B->sq[RAY[s][d][i]];
            if (p == EMPTY) continue;
            if (COLOR(p) == c && (TYPE(p) == slider || TYPE(p) == QUEEN)) return 1;
            break;
        }
    }
    return 0;
}

static void make_move(Board *B, Move m) {
    int c = B->stm, p = B->sq[m.from], t = TYPE(p), cap = B->sq[m.to];
    B->halfmove = (t == PAWN || cap != EMPTY) ? 0 : B->halfmove + 1;
    if (t == PAWN && m.to == B->ep) {  // en passant capture
        B->sq[c == 0 ? m.to - 8 : m.to + 8] = EMPTY;
        B->halfmove = 0;
    }
    B->sq[m.to] = m.promo ? MK(c, m.promo) : p;
    B->sq[m.from] = EMPTY;
    if (t == KING) {
        B->ksq[c] = m.to;
        if (m.to - m.from == 2) { B->sq[m.from + 1] = B->sq[m.from + 3]; B->sq[m.from + 3] = EMPTY; }
        if (m.from - m.to == 2) { B->sq[m.from - 1] = B->sq[m.from - 4]; B->sq[m.from - 4] = EMPTY; }
    }
    B->ep = (t == PAWN && (m.to - m.from == 16 || m.from - m.to == 16)) ? (m.from + m.to) / 2 : -1;
    // castling rights die when a king or rook leaves, or a rook is captured on its corner
    for (int k = 0; k < 2; k++) {
        int s = k ? m.to : m.from;
        if (s == 4) B->castle &= ~3; else if (s == 60) B->castle &= ~12;
        else if (s == 7) B->castle &= ~1; else if (s == 0) B->castle &= ~2;
        else if (s == 63) B->castle &= ~4; else if (s == 56) B->castle &= ~8;
    }
    if (c == 1) B->fullmove++;
    B->stm ^= 1;
}

static int add_pawn(Move *out, int n, int from, int to, int c) {
    if ((c == 0 && RANK_OF(to) == 7) || (c == 1 && RANK_OF(to) == 0)) {
        for (int pr = QUEEN; pr >= KNIGHT; pr--) { out[n].from = from; out[n].to = to; out[n].promo = pr; n++; }
    } else { out[n].from = from; out[n].to = to; out[n].promo = 0; n++; }
    return n;
}

static int gen_pseudo(const Board *B, Move *out) {
    int c = B->stm, n = 0, fwd = c == 0 ? 8 : -8;
    for (int s = 0; s < 64; s++) {
        int p = B->sq[s];
        if (p == EMPTY || COLOR(p) != c) continue;
        int t = TYPE(p);
        if (t == PAWN) {
            int to = s + fwd;
            if (to >= 0 && to < 64 && B->sq[to] == EMPTY) {
                n = add_pawn(out, n, s, to, c);
                int r = RANK_OF(s);
                if (((c == 0 && r == 1) || (c == 1 && r == 6)) && B->sq[to + fwd] == EMPTY) {
                    out[n].from = s; out[n].to = to + fwd; out[n].promo = 0; n++;
                }
            }
            for (int df = -1; df <= 1; df += 2) {
                int f = FILE_OF(s) + df;
                if (f < 0 || f > 7) continue;
                int cs = s + fwd + df;
                if (cs < 0 || cs >= 64) continue;
                int q = B->sq[cs];
                if ((q != EMPTY && COLOR(q) != c) || cs == B->ep) n = add_pawn(out, n, s, cs, c);
            }
        } else if (t == KNIGHT || t == KING) {
            const int8_t *tg = t == KNIGHT ? N_TGT[s] : K_TGT[s];
            int cnt = t == KNIGHT ? N_CNT[s] : K_CNT[s];
            for (int i = 0; i < cnt; i++) {
                int q = B->sq[tg[i]];
                if (q == EMPTY || COLOR(q) != c) { out[n].from = s; out[n].to = tg[i]; out[n].promo = 0; n++; }
            }
        } else {
            int d0 = t == BISHOP ? 4 : 0, d1 = t == ROOK ? 4 : 8;
            for (int d = d0; d < d1; d++) {
                for (int i = 0; i < RAY_LEN[s][d]; i++) {
                    int to = RAY[s][d][i], q = B->sq[to];
                    if (q != EMPTY && COLOR(q) == c) break;
                    out[n].from = s; out[n].to = to; out[n].promo = 0; n++;
                    if (q != EMPTY) break;
                }
            }
        }
    }
    // castling
    int o = c ? 56 : 0, e = c ^ 1;
    if (B->sq[o + 4] == MK(c, KING) && !attacked(B, o + 4, e)) {
        if ((B->castle & (c ? 4 : 1)) && B->sq[o + 7] == MK(c, ROOK) && B->sq[o + 5] == EMPTY &&
            B->sq[o + 6] == EMPTY && !attacked(B, o + 5, e) && !attacked(B, o + 6, e)) {
            out[n].from = o + 4; out[n].to = o + 6; out[n].promo = 0; n++;
        }
        if ((B->castle & (c ? 8 : 2)) && B->sq[o] == MK(c, ROOK) && B->sq[o + 1] == EMPTY &&
            B->sq[o + 2] == EMPTY && B->sq[o + 3] == EMPTY && !attacked(B, o + 3, e) && !attacked(B, o + 2, e)) {
            out[n].from = o + 4; out[n].to = o + 2; out[n].promo = 0; n++;
        }
    }
    return n;
}

static int gen_legal(const Board *B, Move *out) {
    Move ps[256];
    int np = gen_pseudo(B, ps), n = 0, c = B->stm;
    for (int i = 0; i < np; i++) {
        Board b2 = *B;
        make_move(&b2, ps[i]);
        if (!attacked(&b2, b2.ksq[c], c ^ 1)) out[n++] = ps[i];
    }
    return n;
}

static inline int promo_code(int pt) { return pt == QUEEN ? 1 : pt == ROOK ? 2 : pt == BISHOP ? 3 : pt == KNIGHT ? 4 : 0; }
static inline int promo_type(int code) { static const int8_t m[5] = {0, QUEEN, ROOK, BISHOP, KNIGHT}; return m[code]; }
static inline uint16_t pack(Move m) { return (uint16_t)(m.from | (m.to << 6) | (promo_code(m.promo) << 12)); }
static inline Move unpack(uint16_t v) { Move m; m.from = v & 63; m.to = (v >> 6) & 63; m.promo = promo_type(v >> 12); return m; }

// ----------------------------------------------------------------------------------- SAN parsing
static int parse_san(const Board *B, const char *s, int len, Move *res) {
    while (len > 0 && (s[len - 1] == '+' || s[len - 1] == '#' || s[len - 1] == '!' || s[len - 1] == '?')) len--;
    if (len < 2) return 0;
    Move legal[256];
    int n = gen_legal(B, legal), c = B->stm;
    if (s[0] == 'O' || s[0] == '0') {
        int qs = len >= 5;
        int from = c ? 60 : 4, to = qs ? from - 2 : from + 2;
        for (int i = 0; i < n; i++)
            if (legal[i].from == from && legal[i].to == to && TYPE(B->sq[from]) == KING) { *res = legal[i]; return 1; }
        return 0;
    }
    int pt = PAWN, i = 0;
    switch (s[0]) {
        case 'N': pt = KNIGHT; i = 1; break; case 'B': pt = BISHOP; i = 1; break;
        case 'R': pt = ROOK; i = 1; break; case 'Q': pt = QUEEN; i = 1; break; case 'K': pt = KING; i = 1; break;
    }
    int promo = 0, end = len;
    for (int k = i; k < len; k++) {
        if (s[k] == '=' || (pt == PAWN && k > i && (s[k] == 'Q' || s[k] == 'R' || s[k] == 'B' || s[k] == 'N'))) {
            char pc = s[k] == '=' ? (k + 1 < len ? s[k + 1] : 0) : s[k];
            promo = pc == 'Q' ? QUEEN : pc == 'R' ? ROOK : pc == 'B' ? BISHOP : pc == 'N' ? KNIGHT : -1;
            if (promo < 0) return 0;
            end = k; break;
        }
    }
    // destination = last file+rank pair before `end`
    if (end - i < 2) return 0;
    int tf = s[end - 2] - 'a', tr = s[end - 1] - '1';
    if (tf < 0 || tf > 7 || tr < 0 || tr > 7) return 0;
    int to = tr * 8 + tf, ff = -1, fr = -1;
    for (int k = i; k < end - 2; k++) {
        if (s[k] >= 'a' && s[k] <= 'h') ff = s[k] - 'a';
        else if (s[k] >= '1' && s[k] <= '8') fr = s[k] - '1';
    }
    int found = -1;
    for (int k = 0; k < n; k++) {
        Move m = legal[k];
        if (m.to != to || TYPE(B->sq[m.from]) != pt || m.promo != promo) continue;
        if (ff >= 0 && FILE_OF(m.from) != ff) continue;
        if (fr >= 0 && RANK_OF(m.from) != fr) continue;
        if (found >= 0) return 0;  // ambiguous
        found = k;
    }
    if (found < 0) return 0;
    *res = legal[found];
    (void)c;
    return 1;
}

// Parse n games of Lichess movetext. Game g is buf[offs[g]:offs[g+1]].
// Output for each parsed ply: packed move and clock after the move in seconds (-1 if absent).
// nply[g] = plies written for game g, or -1 if the game failed to parse (its plies are not written).
// Returns total plies written, or -2 if `cap` would overflow.
int64_t cb_parse_games(const char *buf, const int64_t *offs, int n, uint16_t *out_moves, int32_t *out_clk,
                       int32_t *nply, int64_t cap, int max_ply) {
    init_tables();
    int64_t w = 0;
    for (int g = 0; g < n; g++) {
        const char *p = buf + offs[g], *e = buf + offs[g + 1];
        Board B; board_init(&B);
        int64_t start = w; int ok = 1, k = 0;
        while (p < e && ok) {
            char ch = *p;
            if (ch == ' ' || ch == '\n' || ch == '\r' || ch == '\t') { p++; continue; }
            if (ch == '{') {
                const char *q = p + 1;
                while (q < e && *q != '}') q++;
                const char *cl = NULL;
                for (const char *r = p; r + 5 < q; r++) if (r[0] == '%' && r[1] == 'c' && r[2] == 'l' && r[3] == 'k') { cl = r + 4; break; }
                if (cl && w > start) {
                    while (cl < q && *cl == ' ') cl++;
                    int hh = 0, mm = 0, ss = 0;
                    while (cl < q && *cl >= '0' && *cl <= '9') hh = hh * 10 + (*cl++ - '0');
                    if (cl < q && *cl == ':') cl++;
                    while (cl < q && *cl >= '0' && *cl <= '9') mm = mm * 10 + (*cl++ - '0');
                    if (cl < q && *cl == ':') cl++;
                    while (cl < q && *cl >= '0' && *cl <= '9') ss = ss * 10 + (*cl++ - '0');
                    out_clk[w - 1] = hh * 3600 + mm * 60 + ss;
                }
                p = q < e ? q + 1 : e;
                continue;
            }
            const char *q = p;
            while (q < e && *q != ' ' && *q != '\n' && *q != '{') q++;
            int len = (int)(q - p);
            if (ch >= '0' && ch <= '9') {
                int dot = 0;
                for (int r = 0; r < len; r++) if (p[r] == '.') dot = 1;
                if (dot) { p = q; continue; }                                   // move number
                if (!(len >= 3 && p[0] == '0' && p[1] == '-' && p[2] == '0')) break;  // result token
            }
            if (ch == '*' || ch == '$') { p = q; continue; }
            Move m;
            if (!parse_san(&B, p, len, &m)) { ok = 0; break; }
            if (k >= max_ply) { break; }
            if (w >= cap) return -2;
            out_moves[w] = pack(m); out_clk[w] = -1; w++; k++;
            make_move(&B, m);
            p = q;
        }
        if (!ok) { w = start; nply[g] = -1; } else nply[g] = (int32_t)(w - start);
    }
    return w;
}

// ----------------------------------------------------------------------------------- batches
static inline int persp(int s, int black) { return black ? s ^ 56 : s; }

static inline int vocab_index(Move m, int black) {
    int f = persp(m.from, black), t = persp(m.to, black);
    if (m.promo) {
        int pi = m.promo == QUEEN ? 0 : m.promo == ROOK ? 1 : m.promo == BISHOP ? 2 : 3;
        return 4096 + (FILE_OF(f) * 8 + FILE_OF(t)) * 4 + pi;
    }
    return f * 64 + t;
}

// Replays each sampled game up to `ply` and emits, from the mover's perspective
// (board flipped vertically and colours swapped when black is to move):
//   board  int8[B,64]   0 empty, 1..6 own P N B R Q K, 7..12 opponent
//   meta   int16[B,8]   own K-castle, own Q-castle, opp K, opp Q, ep square (-1), halfmove, fullmove, in_check
//   hist   int8[B,H,4]  j-th previous ply (j=1..H): from, to, moved piece type 0..5, capture flag; -1 if none
//   legal  uint8[B,4352/8] little-endian bitmask over the 4352-move vocabulary (NULL to skip)
//   target int16[B]     vocabulary index of the move played at `ply` (-1 if ply == game length)
// Returns 0, or 1 + index of the first sample whose stored game holds an illegal move.
int cb_build_batch(const uint16_t *moves, const int64_t *goff, const int64_t *gidx, const int32_t *ply, int nb,
                   int H, int8_t *board, int16_t *meta, int8_t *hist, uint8_t *legal, int16_t *target) {
    init_tables();
    int8_t ptype[1024], pcap[1024];
    for (int b = 0; b < nb; b++) {
        int64_t g = gidx[b], o = goff[g];
        int n = (int)(goff[g + 1] - o), P = ply[b];
        if (P > n || P > 1023) return 1 + b;
        Board B; board_init(&B);
        for (int k = 0; k < P; k++) {
            Move m = unpack(moves[o + k]);
            int pc = B.sq[m.from];
            if (pc == EMPTY || COLOR(pc) != B.stm) return 1 + b;
            ptype[k] = TYPE(pc) - 1;
            pcap[k] = B.sq[m.to] != EMPTY || (TYPE(pc) == PAWN && m.to == B.ep);
            make_move(&B, m);
        }
        int black = B.stm, c = B.stm;
        int8_t *bo = board + (int64_t)b * 64;
        for (int s = 0; s < 64; s++) {
            int p = B.sq[persp(s, black)];
            bo[s] = p == EMPTY ? 0 : (COLOR(p) == c ? TYPE(p) : 6 + TYPE(p));
        }
        int16_t *me = meta + (int64_t)b * 8;
        int ownK = c ? 4 : 1, ownQ = c ? 8 : 2, oppK = c ? 1 : 4, oppQ = c ? 2 : 8;
        me[0] = !!(B.castle & ownK); me[1] = !!(B.castle & ownQ); me[2] = !!(B.castle & oppK); me[3] = !!(B.castle & oppQ);
        me[4] = B.ep >= 0 ? persp(B.ep, black) : -1;
        me[5] = B.halfmove; me[6] = B.fullmove; me[7] = attacked(&B, B.ksq[c], c ^ 1);
        int8_t *hi = hist + (int64_t)b * H * 4;
        for (int j = 1; j <= H; j++) {
            int8_t *h = hi + (j - 1) * 4;
            int k = P - j;
            if (k < 0) { h[0] = h[1] = h[2] = h[3] = -1; continue; }
            Move m = unpack(moves[o + k]);
            h[0] = persp(m.from, black); h[1] = persp(m.to, black); h[2] = ptype[k]; h[3] = pcap[k];
        }
        if (legal) {
            uint8_t *lg = legal + (int64_t)b * 544;
            memset(lg, 0, 544);
            Move lm[256];
            int nl = gen_legal(&B, lm);
            for (int i = 0; i < nl; i++) { int v = vocab_index(lm[i], black); lg[v >> 3] |= (uint8_t)(1 << (v & 7)); }
        }
        if (target) {
            if (P < n) {
                Move m = unpack(moves[o + P]);
                target[b] = (int16_t)vocab_index(m, black);
            } else target[b] = -1;
        }
    }
    return 0;
}

// Maia-3 teacher inputs: for each sample the last `H` positions (oldest first, earliest padded),
// each as 12 bitboards mirrored to *its own* side to move, exactly like maia3.dataset.tokenize_board.
int cb_build_maia(const uint16_t *moves, const int64_t *goff, const int64_t *gidx, const int32_t *ply, int nb,
                  int H, uint64_t *out) {
    init_tables();
    for (int b = 0; b < nb; b++) {
        int64_t g = gidx[b], o = goff[g];
        int P = ply[b];
        Board B; board_init(&B);
        uint64_t *ob = out + (int64_t)b * H * 12;
        for (int k = 0; k <= P; k++) {
            // position before ply k is history slot (H-1) - (P-k)
            int slot = (H - 1) - (P - k);
            if (slot >= 0) {
                uint64_t *bb = ob + slot * 12;
                memset(bb, 0, 12 * sizeof(uint64_t));
                int black = B.stm;
                for (int s = 0; s < 64; s++) {
                    int p = B.sq[s];
                    if (p == EMPTY) continue;
                    int own = COLOR(p) == B.stm;
                    int plane = (own ? 0 : 6) + TYPE(p) - 1;
                    bb[plane] |= 1ULL << persp(s, black);
                }
                if (k == 0) for (int z = 0; z < slot; z++) memcpy(ob + z * 12, bb, 12 * sizeof(uint64_t));
            }
            if (k < P) make_move(&B, unpack(moves[o + k]));
        }
    }
    return 0;
}

// Legal moves of the position after `nply` plies, packed like the store (for engines and tests).
int cb_legal_after(const uint16_t *mv, int nply, uint16_t *out) {
    init_tables();
    Board B; board_init(&B);
    for (int k = 0; k < nply; k++) make_move(&B, unpack(mv[k]));
    Move lm[256];
    int n = gen_legal(&B, lm);
    for (int i = 0; i < n; i++) out[i] = pack(lm[i]);
    return n;
}
