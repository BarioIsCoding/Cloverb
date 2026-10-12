"""Fast Maia-3 inference for distillation: same weights and math as maia3.models.MAIA3Model,
but with fused QKV + SDPA attention, a vectorised promotion head and no per-token Elo concat
(the Elo embeddings enter through the token projection as a per-position bias)."""
from __future__ import annotations

import math
import sys
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/user/ext/maia3")

MODELS = "/home/user/models/maia3"
CONFIGS = {
    "maia3-5m": ("maia3-5m.pt", dict(d=256, h=8, gen=64, psq=0, inter=64)),
    "maia3-23m": ("maia3-23m.pt", dict(d=512, h=16, gen=128, psq=32, inter=128)),
    "maia3-79m": ("maia3-79m.pt", dict(d=1024, h=32, gen=128, psq=32, inter=128)),
}


class FastMaia3(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        fname, c = CONFIGS[name]
        sd = torch.load(f"{MODELS}/{fname}", map_location="cpu", weights_only=True)
        sd = {k.replace("smolgen", "gab"): v.float() for k, v in sd.items()}
        self.c = c
        d, L = c["d"], 8
        self.H = 8
        self.wtok = nn.Parameter(sd["token_projection.weight"][:, :96].clone())
        we = sd["token_projection.weight"][:, 96:]                      # d, 256 (self 128, oppo 128)
        lo, hi = sd["elo_embedding_low.weight"][0], sd["elo_embedding_high.weight"][0]
        # interpolate_elo(e) = (e/5000)*lo + (1 - e/5000)*hi  ->  token bias is affine in each Elo
        self.register_buffer("b0", sd["token_projection.bias"] + we[:, :128] @ hi + we[:, 128:] @ hi)
        self.register_buffer("bs", we[:, :128] @ (lo - hi) / 5000)
        self.register_buffer("bo", we[:, 128:] @ (lo - hi) / 5000)
        self.register_buffer("templates", sd["gab_shared_weight"].T.contiguous())   # gen, 4096
        self.layers = nn.ModuleList()

        def lin(w, b=None):
            m = nn.Linear(w.shape[1], w.shape[0], bias=b is not None)
            m.weight = nn.Parameter(w)
            if b is not None:
                m.bias = nn.Parameter(b)
            return m

        for i in range(L):
            p = f"transformer.layers.{i}."
            m = nn.Module()
            m.qkv = lin(sd[p + "self_attn.mha.in_proj_weight"])
            m.out = lin(sd[p + "self_attn.mha.out_proj.weight"], sd.get(p + "self_attn.mha.out_proj.bias"))
            for k in ("sm1", "sm2", "sm3"):
                if p + f"self_attn.{k}.weight" in sd:
                    setattr(m, k, lin(sd[p + f"self_attn.{k}.weight"], sd[p + f"self_attn.{k}.bias"]))
            m.ln1 = nn.LayerNorm(sd[p + "self_attn.ln1.weight"].shape[0])
            m.ln1.weight = nn.Parameter(sd[p + "self_attn.ln1.weight"]); m.ln1.bias = nn.Parameter(sd[p + "self_attn.ln1.bias"])
            m.ln2 = nn.LayerNorm(sd[p + "self_attn.ln2.weight"].shape[0])
            m.ln2.weight = nn.Parameter(sd[p + "self_attn.ln2.weight"]); m.ln2.bias = nn.Parameter(sd[p + "self_attn.ln2.bias"])
            m.fc1 = lin(sd[p + "linear1.weight"], sd[p + "linear1.bias"])
            m.fc2 = lin(sd[p + "linear2.weight"], sd[p + "linear2.bias"])
            m.n1 = nn.Parameter(sd[p + "norm1.weight"]); m.n2 = nn.Parameter(sd[p + "norm2.weight"])
            self.layers.append(m)
        self.fn = (sd["transformer.norm.weight"], sd["transformer.norm.bias"])
        self.pf = nn.Parameter(sd["proj_sq_from.weight"]); self.pt = nn.Parameter(sd["proj_sq_to.weight"])
        self.pp = nn.Parameter(sd["promo_bias_proj.weight"])
        self.d = d
        self.requires_grad_(False)
        self.eval()

    def gab(self, m, x):
        B = x.shape[0]
        if hasattr(m, "sm1"):
            y = F.gelu(m.sm2(m.sm1(x).reshape(B, -1)))
        else:
            y = F.gelu(m.sm2(x.mean(1)))
        y = F.gelu(m.sm3(m.ln1(y)))
        y = m.ln2(y).view(B, self.c["h"], self.c["gen"])
        return (y @ self.templates).view(B, self.c["h"], 64, 64)

    def forward(self, x, self_elo, oppo_elo):
        """x float[B,64,96] Maia-3 planes (8 boards), Elo long/float [B] -> logits float[B,4352]."""
        B, d, h = x.shape[0], self.d, self.c["h"]
        se, oe = self_elo.float().clamp(0, 5000), oppo_elo.float().clamp(0, 5000)
        x = x @ self.wtok.T + self.b0 + se[:, None, None] * self.bs + oe[:, None, None] * self.bo
        for m in self.layers:
            bias = self.gab(m, x).to(x.dtype)
            q, k, v = m.qkv(x).view(B, 64, 3, h, d // h).permute(2, 0, 3, 1, 4)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
            a = m.out(a.transpose(1, 2).reshape(B, 64, d))
            x = F.rms_norm(x + a, (d,), m.n1)
            x = F.rms_norm(x + m.fc2(F.gelu(m.fc1(x))), (d,), m.n2)
        x = F.layer_norm(x, (d,), *self.fn)
        with torch.autocast("cpu", enabled=False):
            x = x.float()
            f, t = x @ self.pf.T, x @ self.pt.T
            base = f @ t.transpose(1, 2) / math.sqrt(d)
            promo = (t[:, 56:64] @ self.pp.T) * math.sqrt(d)
            pl = (base[:, 48:56, 56:64].unsqueeze(-1) + promo.unsqueeze(1)).reshape(B, 256)
            return torch.cat([base.reshape(B, 4096), pl], 1)


def teacher_logprobs(model, x, se, oe, legal: torch.Tensor, bf16=True):
    """Legal-masked log-probabilities of a Maia-3 teacher."""
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        lo = model(x, se, oe)
    lo = lo.float().masked_fill(~legal, float("-inf"))
    return torch.log_softmax(lo, 1)
