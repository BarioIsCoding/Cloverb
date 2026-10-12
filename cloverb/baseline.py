"""Chessformer / Maia-3 baseline for equal-budget comparisons, built from the authors' released model code.

Inputs follow maia3.uci exactly: the last 8 boards (each mirrored to its own side to move), self and
opponent Elo; no clock. The policy head's promotion logits are computed with the same math as
MAIA3Model.forward but vectorised (the released code loops 256 times in Python).
"""
from __future__ import annotations

import math
import sys
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/user/ext/maia3")
from maia3.models import MAIA3Model  # noqa: E402

from . import core  # noqa: E402

SIZES = {
    "3m": dict(dim_vit=192, head_hid_dim=192, num_heads=6, gab_gen_size=64, gab_per_square_dim=0, gab_intermediate_dim=64),
    "5m": dict(dim_vit=256, head_hid_dim=256, num_heads=8, gab_gen_size=64, gab_per_square_dim=0, gab_intermediate_dim=64),
}
BASE = dict(history=8, use_padding=True, include_time_info=False, dim_emb=128, num_blocks=8, mlp_ratio=2.0,
            dropout=0.0, use_gab=True, use_relative_bias=False, use_absolute_pe=False, use_rms_norm=True,
            omit_qkv_biases=True, activation="gelu")


class Chessformer(nn.Module):
    def __init__(self, size="5m"):
        super().__init__()
        self.cfg = Namespace(**BASE, **SIZES[size])
        self.m = MAIA3Model(self.cfg)

    def forward(self, x, self_elo, oppo_elo):
        m, cfg = self.m, self.cfg
        se = m.interpolate_elo(self_elo).unsqueeze(1).expand(-1, 64, -1)
        oe = m.interpolate_elo(oppo_elo).unsqueeze(1).expand(-1, 64, -1)
        h = m.transformer(m.token_projection(torch.cat([x, se, oe], -1)))
        if not self.training:  # fp32 head at eval (bf16 logit ties inflate top-1)
            with torch.autocast("cpu", enabled=False):
                return self._head(h.float())
        return self._head(h)

    def _head(self, h):
        m, cfg = self.m, self.cfg
        f, t = m.proj_sq_from(h), m.proj_sq_to(h)
        d = cfg.head_hid_dim
        base = f @ t.transpose(1, 2) / math.sqrt(d)
        promo = m.promo_bias_proj(t[:, 56:64]) * math.sqrt(d)
        B = h.shape[0]
        pl = (base[:, 48:56, 56:64].unsqueeze(-1) + promo.unsqueeze(1)).reshape(B, 256)
        logits = torch.cat([base.reshape(B, 4096), pl], 1)
        g = m.last_ln(h.mean(1))
        return logits, m.fc_value(F.relu(m.fc_value_hid(g)))


def maia_inputs(gs, g, ply, H=8):
    bb = core.build_maia(gs.moves, gs.offsets, g, ply, H)                     # B,H,12 uint64
    B = len(g)
    bits = np.unpackbits(bb.view(np.uint8), axis=-1, bitorder="little").reshape(B, H, 12, 64)
    x = torch.from_numpy(bits.transpose(0, 3, 1, 2).reshape(B, 64, H * 12)).float()
    white = ply % 2 == 0
    se = np.where(white, gs.welo[g], gs.belo[g]).astype(np.int64)
    oe = np.where(white, gs.belo[g], gs.welo[g]).astype(np.int64)
    return x, torch.from_numpy(se), torch.from_numpy(oe)
