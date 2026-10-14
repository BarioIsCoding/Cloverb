"""Cloverb: a square-token transformer for human move prediction.

Differences from Chessformer / Maia-3 (see docs/critique.md for the reasoning):
  * inputs are compact integers, embedded in-graph: the current board (13-way piece embedding
    + absolute square embedding), castling rights on their rook squares, the en-passant square,
    and the last <= 64 plies as *moves* (from/to squares, moved piece, capture) scatter-added onto
    the squares they touched, all in one consistent mover-relative orientation;
  * conditioning is a single vector built from the player's Elo, the time control (base,
    increment), the time left and game counters; it is added to every token and modulates every
    block through adaptive RMSNorm (adaLN-Zero) instead of being concatenated to all 64 tokens;
  * pre-norm blocks with QK-norm (trainable at ~20x Maia-3's learning rate), fused SDPA attention;
  * attention bias = per-layer static 64x64 bias per head + a cheap dynamic part mixing
    board-conditioned coefficients over templates shared by all layers (GAB-lite);
  * vectorised from-to policy head over the Maia-3 4352-move vocabulary (so Maia-3 can be used
    as a teacher), plus auxiliary WDL and think-time heads.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # fused AVX-512 attention for 64-token boards (csrc/attn64.c); SDPA fallback elsewhere
    with open("/proc/cpuinfo") as _f:
        _HAS_AVX512 = "avx512f" in _f.read()
    if _HAS_AVX512:
        from .attn import attn64, attn64qkv
    FUSED_ATTN = _HAS_AVX512
except Exception:  # pragma: no cover
    FUSED_ATTN = False

FUSED_QKV = True  # norm/permute/transpose fused into the attention kernel (csrc/attn64.c)
QK_EPS = None     # override the q/k RMSNorm eps (default: finfo of the activation dtype, 2^-7 under bf16)

VOCAB = 4352
ELO_ANCHORS = torch.arange(400.0, 3201.0, 200.0)          # 15 hat functions
LOGT_ANCHORS = torch.log1p(torch.tensor([2., 5, 10, 20, 30, 45, 60, 90, 120, 180, 300, 600, 1200, 3600]))
TIME_BINS = 32                                             # think-time head: log-spaced buckets


@dataclass
class CloverbConfig:
    d: int = 256
    layers: int = 8
    heads: int = 8
    mlp_mult: float = 3.0
    hist: int = 64
    d_cond: int = 128
    n_templates: int = 32          # dynamic attention-bias templates (0 disables)
    static_bias: bool = True
    adaln: bool = True
    value_hidden: int = 128
    act: str = "relu2"             # MLP activation: relu2 | gelu
    geo_init: bool = False         # initialise static attention biases with chess move patterns
    opp_elo: bool = False          # diagnostic only: also condition on the opponent's Elo
    zero_init: bool = True         # zero-init residual output projections (else small normal init)
    cap_type: bool = False         # history knows which piece was captured (else only that one was)
    postln: bool = False           # post-norm blocks (norm after each residual add), as in Chessformer

    def to_dict(self):
        return asdict(self)


def hat(x: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear (hat-function) encoding of a scalar on fixed anchors, clamped at the ends."""
    a = anchors.to(x.device, x.dtype)
    x = x.clamp(a[0], a[-1]).unsqueeze(-1)
    big = torch.ones(1, device=x.device, dtype=x.dtype) * 1e9
    wl = torch.cat([big, a[1:] - a[:-1]])      # distance to the previous anchor
    wr = torch.cat([a[1:] - a[:-1], big])      # distance to the next anchor
    return torch.where(x <= a, 1 - (a - x) / wl, 1 - (x - a) / wr).clamp(0, 1)


N_COND = 15 + 14 + 6 + 4 + 2
N_COND_OPP = N_COND + 15


def cond_features(elo, base, inc, clock, meta, hist_len):
    """Global conditioning features. elo/base/inc/clock in rating points and seconds."""
    elo, base, inc, clock = (t.float() for t in (elo, base, inc, clock))
    lt = torch.log1p(clock)
    full = meta[:, 6].float()
    return torch.cat([
        hat(elo, ELO_ANCHORS),
        hat(lt, LOGT_ANCHORS),
        torch.stack([
            torch.log1p(base) / 8, torch.log1p(inc) / 4, lt / 8,
            (clock / (base + 40 * inc + 1)).clamp(0, 2),
            meta[:, 5].float().clamp(max=100) / 100,                 # halfmove clock
            full.clamp(max=150) / 100,                               # fullmove number
        ], -1),
        meta[:, 0:4].float(),                                        # castling rights (also on squares)
        torch.stack([meta[:, 7].float(), hist_len.float() / 64], -1),
    ], -1)


class RMSNorm(nn.Module):
    def __init__(self, d, affine=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d)) if affine else None
        self.d = d

    def forward(self, x):
        return F.rms_norm(x, (self.d,), self.weight)


def geometry_bias(h: int, strength: float = 2.0) -> torch.Tensor:
    """Per-head 64x64 attention-bias prior from chess geometry (query square i attends key square j):
    rook lines, bishop diagonals, knight jumps, king neighbourhood, pawn-attack squares, same file,
    Chebyshev-distance decay; remaining heads start at zero."""
    sq = torch.arange(64)
    f, r = sq % 8, sq // 8
    df, dr = f[None, :] - f[:, None], r[None, :] - r[:, None]
    adf, adr = df.abs(), dr.abs()
    off = (adf + adr) > 0
    pats = [
        ((adf == 0) | (adr == 0)) & off,                       # rook / queen lines
        (adf == adr) & off,                                    # bishop / queen diagonals
        ((adf == 1) & (adr == 2)) | ((adf == 2) & (adr == 1)), # knight
        (torch.maximum(adf, adr) == 1),                        # king
        (adf == 1) & (dr.abs() == 1),                          # pawn attacks (both directions)
        (adf == 0) & off,                                      # same file (pawn structure)
    ]
    out = torch.zeros(h, 64, 64)
    for k in range(min(h, len(pats))):
        out[k] = strength * pats[k].float()
    if h > len(pats):
        out[len(pats)] = -0.5 * torch.maximum(adf, adr).float()   # locality
    return out


class Block(nn.Module):
    def __init__(self, cfg: CloverbConfig):
        super().__init__()
        d, h = cfg.d, cfg.heads
        self.h, self.dh = h, d // h
        self.norm1 = RMSNorm(d, affine=not cfg.adaln)
        self.norm2 = RMSNorm(d, affine=not cfg.adaln)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.q_norm = RMSNorm(self.dh)
        self.k_norm = RMSNorm(self.dh)
        self.proj = nn.Linear(d, d, bias=False)
        hid = int(cfg.mlp_mult * d)
        self.fc1 = nn.Linear(d, hid, bias=False)
        self.fc2 = nn.Linear(hid, d, bias=False)
        self.adaln = nn.Linear(cfg.d_cond, 6 * d) if cfg.adaln else None
        if self.adaln is not None:
            nn.init.zeros_(self.adaln.weight); nn.init.zeros_(self.adaln.bias)
        self.static = nn.Parameter(torch.zeros(h, 64, 64)) if cfg.static_bias else None
        if cfg.n_templates:
            self.dyn = nn.Linear(cfg.d + cfg.d_cond, h * cfg.n_templates)
            nn.init.zeros_(self.dyn.weight); nn.init.zeros_(self.dyn.bias)
        else:
            self.dyn = None
        if cfg.zero_init:
            nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.fc2.weight)
        else:
            nn.init.normal_(self.proj.weight, std=0.02 / math.sqrt(2 * cfg.layers))
            nn.init.normal_(self.fc2.weight, std=0.02 / math.sqrt(2 * cfg.layers))
        self.act = (lambda t: F.relu(t).square()) if cfg.act == "relu2" else F.gelu
        self.postln = cfg.postln
        if cfg.postln:  # post-norm needs affine norms; adaLN still modulates the branch inputs
            self.norm1, self.norm2 = RMSNorm(d), RMSNorm(d)
        if cfg.geo_init and self.static is not None:
            with torch.no_grad():
                self.static.copy_(geometry_bias(h))

    def bias(self, x, c, templates):
        B = x.shape[0]
        out = None
        if self.dyn is not None:
            coef = self.dyn(torch.cat([x.mean(1), c], -1)).view(B, self.h, -1)          # B,H,T
            out = (coef @ templates).view(B, self.h, 64, 64)                              # B,H,64,64
        if self.static is not None:
            out = self.static.unsqueeze(0) if out is None else out + self.static
        return out

    def attend(self, h, x, c, templates):
        """Attention branch output (after the output projection), fused kernel or PyTorch fallback."""
        B = h.shape[0]
        coef = (self.dyn(torch.cat([x.mean(1), c], -1)).view(B, self.h, -1) if self.dyn is not None
                else x.new_zeros(B, self.h, 0, dtype=torch.float32))
        if FUSED_ATTN and h.device.type == "cpu":  # the AVX-512 kernel is CPU-only
            qkv = self.qkv(h)
            stat = self.static if self.static is not None else x.new_zeros(self.h, 64, 64, dtype=torch.float32)
            tm = templates if (templates is not None and self.dyn is not None) else x.new_zeros(0, 4096, dtype=torch.float32)
            eps = QK_EPS if QK_EPS is not None else torch.finfo(qkv.dtype).eps
            return self.proj(attn64qkv(qkv, self.q_norm.weight, self.k_norm.weight, stat, coef, tm, self.h,
                                       self.dh ** -0.5, eps))
        q, k, v = self.qkv(h).view(B, 64, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        bias = self.bias(x, c, templates)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=None if bias is None else bias.to(q.dtype))
        return self.proj(a.transpose(1, 2).reshape(B, 64, -1))

    def forward(self, x, c, templates):
        B = x.shape[0]
        if self.postln:  # x is already normalised; modulate the branch inputs, normalise after each add
            if self.adaln is not None:
                s1, b1, g1, s2, b2, g2 = self.adaln(c).unsqueeze(1).chunk(6, -1)
                h = x * (1 + s1) + b1
            else:
                h, g1, g2 = x, 0, 0
            x = self.norm1(x + self.attend(h, x, c, templates) * (1 + g1))
            h = x * (1 + s2) + b2 if self.adaln is not None else x
            return self.norm2(x + self.fc2(self.act(self.fc1(h))) * (1 + g2))
        if self.adaln is not None:
            s1, b1, g1, s2, b2, g2 = self.adaln(c).unsqueeze(1).chunk(6, -1)
            h = self.norm1(x) * (1 + s1) + b1
        else:
            h = self.norm1(x)
        if FUSED_ATTN and FUSED_QKV and h.device.type == "cpu":
            qkv = self.qkv(h)
            coef = (self.dyn(torch.cat([x.mean(1), c], -1)).view(B, self.h, -1) if self.dyn is not None
                    else x.new_zeros(B, self.h, 0, dtype=torch.float32))
            stat = self.static if self.static is not None else x.new_zeros(self.h, 64, 64, dtype=torch.float32)
            tm = templates if (templates is not None and self.dyn is not None) else x.new_zeros(0, 4096, dtype=torch.float32)
            eps = torch.finfo(qkv.dtype).eps  # matches F.rms_norm(eps=None) on the unfused path
            a = self.proj(attn64qkv(qkv, self.q_norm.weight, self.k_norm.weight, stat, coef, tm, self.h,
                                    self.dh ** -0.5, eps))
            x = x + (a * (1 + g1) if self.adaln is not None else a)
            h = self.norm2(x) * (1 + s2) + b2 if self.adaln is not None else self.norm2(x)
            m = self.fc2(self.act(self.fc1(h)))
            return x + (m * (1 + g2) if self.adaln is not None else m)
        q, k, v = self.qkv(h).view(B, 64, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        if FUSED_ATTN and q.device.type == "cpu":
            coef = (self.dyn(torch.cat([x.mean(1), c], -1)).view(B, self.h, -1) if self.dyn is not None
                    else x.new_zeros(B, self.h, 0, dtype=torch.float32))
            stat = self.static if self.static is not None else x.new_zeros(self.h, 64, 64, dtype=torch.float32)
            tm = templates if (templates is not None and self.dyn is not None) else x.new_zeros(0, 4096, dtype=torch.float32)
            a = attn64(q, k, v, stat, coef, tm, self.dh ** -0.5)
        else:
            bias = self.bias(x, c, templates)
            if bias is not None:
                bias = bias.to(q.dtype)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        a = self.proj(a.transpose(1, 2).reshape(B, 64, -1))
        x = x + (a * (1 + g1) if self.adaln is not None else a)
        h = self.norm2(x) * (1 + s2) + b2 if self.adaln is not None else self.norm2(x)
        m = self.fc2(self.act(self.fc1(h)))
        return x + (m * (1 + g2) if self.adaln is not None else m)


class Cloverb(nn.Module):
    def __init__(self, cfg: CloverbConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.piece = nn.Embedding(13, d)
        self.square = nn.Parameter(torch.randn(64, d) * 0.02)
        self.castle = nn.Parameter(torch.randn(4, d) * 0.02)
        self.ep = nn.Parameter(torch.randn(d) * 0.02)
        H = cfg.hist
        self.h_from = nn.Embedding(H, d)       # "a piece left this square j plies ago"
        self.h_to = nn.Embedding(H, d)         # "a piece arrived here j plies ago"
        self.h_piece = nn.Embedding(2 * 6 * 2, d)   # (from/to) x (own/opp) x piece type
        self.h_cap = nn.Embedding(2 * 6 if cfg.cap_type else 2, d)   # capture by own/opp (x captured piece)
        for e in (self.piece, self.h_from, self.h_to, self.h_piece, self.h_cap):
            nn.init.normal_(e.weight, std=0.02)
        self.cond = nn.Sequential(nn.Linear(N_COND_OPP if cfg.opp_elo else N_COND, cfg.d_cond), nn.SiLU(), nn.Linear(cfg.d_cond, cfg.d_cond), nn.SiLU())
        self.cond_tok = nn.Linear(cfg.d_cond, d)
        self.templates = nn.Parameter(torch.randn(cfg.n_templates, 64 * 64) * 0.1) if cfg.n_templates else None
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.in_norm = RMSNorm(d) if cfg.postln else None
        self.norm = RMSNorm(d)
        self.p_from = nn.Linear(d, d, bias=False)
        self.p_to = nn.Linear(d, d, bias=False)
        self.p_promo = nn.Linear(d, 4, bias=False)
        self.v_head = nn.Sequential(nn.Linear(d + cfg.d_cond, cfg.value_hidden), nn.SiLU(), nn.Linear(cfg.value_hidden, 3))
        self.t_head = nn.Sequential(nn.Linear(d + cfg.d_cond, cfg.value_hidden), nn.SiLU(), nn.Linear(cfg.value_hidden, TIME_BINS))
        own_sq = torch.tensor([7, 0, 63, 56])  # rook squares carrying own K/Q, opp K/Q castling rights
        self.register_buffer("castle_sq", own_sq, persistent=False)
        self.register_buffer("parity", (torch.arange(H) % 2 == 0).long(), persistent=False)  # j=1 -> opponent

    def embed(self, board, meta, hist):
        """board int8[B,64], meta int16[B,8], hist int8[B,H,4] -> tokens [B,64,d] (float32)."""
        B, H = board.shape[0], self.cfg.hist
        x = self.piece(board.long()) + self.square
        cast = meta[:, :4].float()                                            # B,4
        x = x.index_add(1, self.castle_sq, cast.unsqueeze(-1) * self.castle)  # rights live on rook squares
        ep = meta[:, 4].long()
        has_ep = (ep >= 0).float()
        x = x.scatter_add(1, ep.clamp(min=0)[:, None, None].expand(B, 1, x.shape[-1]),
                          (has_ep[:, None, None] * self.ep))
        hist = hist[:, :H].long()
        valid = (hist[..., 0] >= 0)                                           # B,H
        side = self.parity.expand(B, H)                                       # 1 = opponent's move
        ptype = hist[..., 2].clamp(min=0)
        j = torch.arange(H, device=board.device).expand(B, H)
        vf = self.h_from(j) + self.h_piece(side * 6 + ptype)
        cap = hist[..., 3].clamp(min=0)
        if self.cfg.cap_type:
            capv = (cap > 0).unsqueeze(-1).to(self.h_cap.weight.dtype) * self.h_cap(side * 6 + cap)
        else:
            capv = (cap > 0).unsqueeze(-1).to(self.h_cap.weight.dtype) * self.h_cap(side)
        vt = self.h_to(j) + self.h_piece(12 + side * 6 + ptype) + capv
        vm = valid.unsqueeze(-1).to(vf.dtype)
        idx_f = hist[..., 0].clamp(min=0).unsqueeze(-1).expand(B, H, x.shape[-1])
        idx_t = hist[..., 1].clamp(min=0).unsqueeze(-1).expand(B, H, x.shape[-1])
        x = x.scatter_add(1, idx_f, vf * vm).scatter_add(1, idx_t, vt * vm)
        return x, valid.sum(1)

    def forward(self, board, meta, hist, elo, base, inc, clock, oelo=None):
        x, hlen = self.embed(board, meta, hist)
        feats = cond_features(elo, base, inc, clock, meta, hlen)
        if self.cfg.opp_elo:
            feats = torch.cat([feats, hat(oelo.float(), ELO_ANCHORS)], -1)
        c = self.cond(feats)
        x = x + self.cond_tok(c).unsqueeze(1)
        if self.cfg.postln:
            x = self.in_norm(x)
        for blk in self.blocks:
            x = blk(x, c, self.templates)
        x = self.norm(x)
        return self.heads(x, c)

    def heads(self, x, c):
        if not self.training:  # bf16 logits tie often enough to inflate top-1 (~1 point); use fp32 at eval
            with torch.autocast(x.device.type, enabled=False):
                return self._heads(x.float(), c.float())
        return self._heads(x, c)

    def _heads(self, x, c):
        B, d = x.shape[0], self.cfg.d
        f, t = self.p_from(x), self.p_to(x)
        base = (f @ t.transpose(1, 2)) / math.sqrt(d)                              # B,64,64
        promo = self.p_promo(t[:, 56:64])                                          # B,8 (to file),4
        pl = (base[:, 48:56, 56:64].unsqueeze(-1) + promo.unsqueeze(1)).reshape(B, 256)
        logits = torch.cat([base.reshape(B, 4096), pl], 1)
        g = torch.cat([x.mean(1), c], -1)
        return logits, self.v_head(g), self.t_head(g)


def time_bucket(seconds: torch.Tensor) -> torch.Tensor:
    """Think-time target bucket: 0 s, then log-spaced up to ~20 min."""
    return (torch.log1p(seconds.float()) * (TIME_BINS - 1) / math.log1p(1200)).round().clamp(0, TIME_BINS - 1).long()


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
