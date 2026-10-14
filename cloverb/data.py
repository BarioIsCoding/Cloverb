"""Sampling training positions from game stores and turning them into model inputs."""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

import numpy as np
import torch

from . import core
from .gamestore import GameStore


def concat_stores(paths: list[str]) -> GameStore:
    if len(paths) == 1:
        return GameStore.load(paths[0])
    parts = [GameStore.load(p) for p in paths]
    offs, base = [np.zeros(1, np.int64)], 0
    for g in parts:
        offs.append(g.offsets[1:] + base)
        base += g.offsets[-1]
    cat = lambda k: np.concatenate([getattr(g, k) for g in parts])
    return GameStore(moves=cat("moves"), clock=cat("clock"), spent=cat("spent"), offsets=np.concatenate(offs),
                     welo=cat("welo"), belo=cat("belo"), base=cat("base"), inc=cat("inc"), result=cat("result"))


def tc_class(base: np.ndarray, inc: np.ndarray) -> np.ndarray:
    """0 bullet, 1 blitz, 2 rapid, 3 classical (Lichess estimated-duration rule)."""
    est = base.astype(np.int64) + 40 * inc.astype(np.int64)
    return np.digitize(est, [180, 480, 1500])


@dataclass
class SamplerConfig:
    elo_alpha: float = 0.3                     # weight ~ p(mover elo bin)^-alpha
    tc_weight: tuple = (0.6, 1.0, 1.3, 4.0)    # bullet, blitz, rapid, classical
    opening_weight: float = 0.5                # plies < 10
    seed: int = 0


class Sampler:
    """Draws (game, ply) samples: first a (game, side) pair by Elo/time-control weight, then a ply."""

    def __init__(self, gs: GameStore, cfg: SamplerConfig, holdout_games: np.ndarray | None = None):
        self.gs, self.cfg = gs, cfg
        self.rng = np.random.default_rng(cfg.seed)
        n = gs.n_games
        lens = np.diff(gs.offsets)
        elo = np.concatenate([gs.welo, gs.belo]).astype(np.int64)
        bins = np.clip(elo // 100, 0, 35)
        p = np.bincount(bins, minlength=36).astype(np.float64)
        p /= p.sum()
        w_elo = np.where(p[bins] > 0, p[bins], 1) ** (-cfg.elo_alpha)
        tcw = np.array(cfg.tc_weight)[tc_class(gs.base, gs.inc)]
        n_side = np.concatenate([(lens + 1) // 2, lens // 2]).astype(np.float64)   # plies played by each side
        w = w_elo * np.concatenate([tcw, tcw]) * n_side
        if holdout_games is not None:
            w[holdout_games] = 0; w[holdout_games + n] = 0
        self.cdf = np.cumsum(w / w.sum())
        self.lens = lens

    def sample(self, B: int):
        n = self.gs.n_games
        # oversample, then thin openings by rejection
        k = int(B * 1.3) + 8
        u = self.rng.random(k)
        gsid = np.searchsorted(self.cdf, u, side="right").clip(0, 2 * n - 1)
        g, side = gsid % n, gsid // n
        cnt = (self.lens[g] + 1 - side) // 2
        ply = side + 2 * (self.rng.random(k) * cnt).astype(np.int64)
        keep = (ply >= 10) | (self.rng.random(k) < self.cfg.opening_weight)
        g, ply = g[keep][:B], ply[keep][:B]
        assert len(g) == B
        return g, ply


def make_inputs(gs: GameStore, g: np.ndarray, ply: np.ndarray, H: int = 64, legal: bool = True):
    """Model inputs and targets for samples (game index, ply) of a store."""
    b = core.build_batch(gs.moves, gs.offsets, g, ply, H=H, legal=legal)
    k = gs.offsets[g] + ply
    white = (ply % 2 == 0)
    elo = np.where(white, gs.welo[g], gs.belo[g]).astype(np.float32)
    oelo = np.where(white, gs.belo[g], gs.welo[g]).astype(np.float32)
    res = gs.result[g].astype(np.int64) * np.where(white, 1, -1)          # +1 mover won
    out = dict(
        board=torch.from_numpy(b.board), meta=torch.from_numpy(b.meta), hist=torch.from_numpy(b.hist),
        elo=torch.from_numpy(elo), oelo=torch.from_numpy(oelo), base=torch.from_numpy(gs.base[g].astype(np.float32)),
        inc=torch.from_numpy(gs.inc[g].astype(np.float32)), clock=torch.from_numpy(gs.clock[k].astype(np.float32)),
        target=torch.from_numpy(b.target.astype(np.int64)), wdl=torch.from_numpy(res + 1),
        spent=torch.from_numpy(np.where(ply >= 2, gs.spent[k].astype(np.int32), -1).astype(np.float32)),
    )
    if legal:
        out["legal"] = torch.from_numpy(np.unpackbits(b.legal, axis=1, bitorder="little").view(np.bool_))
    return out


MODEL_KEYS = ("board", "meta", "hist", "elo", "base", "inc", "clock")


class PoolSampler:
    """Epochs over teacher-labelled pools (tools/label_teacher.py): yields (game, ply, soft target)."""

    def __init__(self, paths: list[str], stores: list[str], seed: int = 0, vocab: int = 4352, n_games: int = -1):
        zs = [np.load(p) for p in paths]
        for z in zs:
            if list(z["stores"]) != list(stores):  # e.g. the merged copy of the same stores: compare sizes
                n = sum(len(np.load(s_, mmap_mode="r")["offsets"]) - 1 if not s_.endswith("/") and s_.endswith(".npz")
                        else len(np.load(s_ + "/offsets.npy", mmap_mode="r")) - 1 for s_ in z["stores"])
                assert n == n_games, "pool was labelled against a different store"
        self.g = np.concatenate([z["game"] for z in zs]); self.p = np.concatenate([z["ply"] for z in zs])
        self.idx = np.concatenate([z["idx"] for z in zs])      # int16 [N, k], kept compact
        self.prob = np.concatenate([z["prob"] for z in zs])    # float16 [N, k]; normalised per batch
        self.rng = np.random.default_rng(seed)
        self.order, self.pos, self.vocab, self.epoch = self.rng.permutation(len(self.g)), 0, vocab, 0

    def sample(self, B: int):
        if self.pos + B > len(self.order):
            self.order, self.pos = self.rng.permutation(len(self.g)), 0
            self.epoch += 1
        r = self.order[self.pos:self.pos + B]
        self.pos += B
        pr = self.prob[r].astype(np.float32)
        pr /= pr.sum(1, keepdims=True)
        soft = torch.zeros(B, self.vocab)
        soft.scatter_(1, torch.from_numpy(self.idx[r].astype(np.int64)), torch.from_numpy(pr))
        return self.g[r], self.p[r], soft


class Prefetcher:
    """Background thread building batches (the C builder releases the GIL)."""

    def __init__(self, gs: GameStore, sampler: Sampler, B: int, H: int, depth: int = 6, extra=None, pin: bool = False):
        self.q: queue.Queue = queue.Queue(depth)
        self.stop = False

        def run():
            try:
                while not self.stop:
                    smp = sampler.sample(B)
                    g, ply = smp[0], smp[1]
                    x = make_inputs(gs, g, ply, H)
                    x["g"], x["ply"] = g, ply
                    if len(smp) > 2:
                        x["soft"] = smp[2]
                    if extra is not None:
                        x.update(extra(gs, g, ply))
                    if pin:  # page-locked host memory for asynchronous copies to the GPU
                        x = {k: v.pin_memory() if torch.is_tensor(v) else v for k, v in x.items()}
                    self.q.put(x)
            except BaseException as e:  # surface loader errors in the training thread
                self.q.put(e)
        self.t = threading.Thread(target=run, daemon=True)
        self.t.start()

    def __next__(self):
        x = self.q.get()
        if isinstance(x, BaseException):
            raise x
        return x

    def close(self):
        self.stop = True
