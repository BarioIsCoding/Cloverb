"""Inference wrapper: positions in, human move distribution / WDL / think time out."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from . import core
from .gamestore import decode_uci, encode_uci
from .model import TIME_BINS, Cloverb, CloverbConfig


@dataclass
class Prediction:
    moves: list          # UCI strings, most likely first
    probs: np.ndarray    # matching probabilities
    wdl: np.ndarray      # mover's win/draw/loss probabilities
    think_s: float       # median predicted think time in seconds
    think_dist: np.ndarray


def load(path: str, threads: int | None = None, compile_: bool = False) -> Cloverb:
    st = torch.load(path, map_location="cpu", weights_only=False)
    cfg = CloverbConfig(**st["cfg"])
    m = Cloverb(cfg)
    m.load_state_dict(st["ema"] if "ema" in st else st["model"])
    m.eval().requires_grad_(False)
    if threads:
        torch.set_num_threads(threads)
    return torch.compile(m) if compile_ else m


def bucket_seconds(b: np.ndarray) -> np.ndarray:
    return np.expm1(b * math.log1p(1200) / (TIME_BINS - 1))


class Predictor:
    def __init__(self, model: Cloverb, bf16: bool = True):
        self.model, self.bf16 = model, bf16
        self.H = (model._orig_mod if hasattr(model, "_orig_mod") else model).cfg.hist

    @torch.inference_mode()
    def predict(self, fen: str | None, moves: list[str], elo: float, base: float, inc: float, clock: float,
                top: int = 8) -> Prediction | None:
        enc = core.encode_position(fen, np.array([encode_uci(u) for u in moves], np.uint16), self.H)
        if enc is None or len(enc[4]) == 0:
            return None
        board, meta, hist, legal, legal_moves = enc
        t = lambda a: torch.from_numpy(a)[None]
        f = lambda v: torch.tensor([float(v)])
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=self.bf16):
            lo, v, tm = self.model(t(board), t(meta), t(hist), f(elo), f(base), f(inc), f(clock))
        black = bool(fen and fen.split()[1] == "b") ^ (len(moves) % 2 == 1)
        idx = np.array([core.vocab_index(int(m), black) for m in legal_moves])
        lg = lo[0].float()[torch.from_numpy(idx)]
        p = torch.softmax(lg, 0).numpy()
        order = np.argsort(-p)[:top]
        td = torch.softmax(tm[0].float(), 0).numpy()
        med = bucket_seconds(np.searchsorted(np.cumsum(td), 0.5))
        wdl = torch.softmax(v[0].float(), 0).numpy()[::-1]          # model emits (loss, draw, win)
        return Prediction([decode_uci(int(legal_moves[i])) for i in order], p[order], wdl.copy(), float(med), td)
