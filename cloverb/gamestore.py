"""Compact game store shared by every Cloverb tool.

A store is one ``.npz`` holding whole games, never individual positions:

* ``moves``   uint16[P]  every ply of every game, ``from | to << 6 | promo << 12``
                         (absolute squares, promo 0=none 1=q 2=r 3=b 4=n)
* ``clock``   uint16[P]  seconds left on the mover's clock *before* the ply
* ``spent``   uint16[P]  seconds the mover spent on the ply
* ``offsets`` int64[G+1] game boundaries into the per-ply arrays
* ``welo``, ``belo`` int16[G]; ``base`` int32[G] (s); ``inc`` int16[G] (s); ``result`` int8[G] (+1 white, 0, -1)

Positions are rebuilt by replaying moves, so a store of 1M games is ~200 MB.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

PROMO_CHARS = " qrbn"


def encode_uci(u: str) -> int:
    f = (ord(u[0]) - 97) + 8 * (ord(u[1]) - 49)
    t = (ord(u[2]) - 97) + 8 * (ord(u[3]) - 49)
    p = PROMO_CHARS.index(u[4]) if len(u) > 4 else 0
    return f | (t << 6) | (p << 12)


def decode_uci(m: int) -> str:
    f, t, p = m & 63, (m >> 6) & 63, m >> 12
    s = f"{chr(97 + (f & 7))}{(f >> 3) + 1}{chr(97 + (t & 7))}{(t >> 3) + 1}"
    return s + (PROMO_CHARS[p] if p else "")


@dataclass
class GameStore:
    moves: np.ndarray
    clock: np.ndarray
    spent: np.ndarray
    offsets: np.ndarray
    welo: np.ndarray
    belo: np.ndarray
    base: np.ndarray
    inc: np.ndarray
    result: np.ndarray

    @property
    def n_games(self) -> int:
        return len(self.offsets) - 1

    def game(self, g: int):
        a, b = self.offsets[g], self.offsets[g + 1]
        return self.moves[a:b], self.clock[a:b], self.spent[a:b]

    def save(self, path: str):
        np.savez(path, **{k: getattr(self, k) for k in self.__dataclass_fields__})

    @classmethod
    def load(cls, path: str, mmap: bool = False) -> "GameStore":
        if os.path.isdir(path):  # directory of .npy files: memory-mapped, shared through the page cache
            return cls(**{k: np.load(os.path.join(path, k + ".npy"), mmap_mode="r") for k in cls.__dataclass_fields__})
        z = np.load(path, mmap_mode="r" if mmap else None)
        return cls(**{k: z[k] for k in cls.__dataclass_fields__})

    def save_dir(self, path: str):
        os.makedirs(path, exist_ok=True)
        for k in self.__dataclass_fields__:
            np.save(os.path.join(path, k + ".npy"), np.ascontiguousarray(getattr(self, k)))


def from_allie_jsonl(path: str) -> GameStore:
    """Convert the ALLIE 2022 blitz test set (Zhang et al., 2025) used by the Maia-3 paper."""
    moves, clock, spent, offsets = [], [], [], [0]
    welo, belo, base, inc, result = [], [], [], [], []
    for line in open(path):
        g = json.loads(line)
        uci = g["moves-uci"].split()
        secs = g["moves-seconds"]
        b, i = map(int, g["time-control"].split("+"))
        clk = [b, b]
        for k, u in enumerate(uci):
            p = k & 1
            moves.append(encode_uci(u))
            clock.append(max(0, min(65535, clk[p])))
            spent.append(max(0, min(65535, secs[k])))
            clk[p] = clk[p] - secs[k] + i
        offsets.append(len(moves))
        welo.append(g["white-elo"]); belo.append(g["black-elo"])
        base.append(b); inc.append(i)
        result.append({"1-0": 1, "0-1": -1}.get(g["result"], 0))
    return GameStore(
        moves=np.array(moves, np.uint16), clock=np.array(clock, np.uint16),
        spent=np.array(spent, np.uint16), offsets=np.array(offsets, np.int64),
        welo=np.array(welo, np.int16), belo=np.array(belo, np.int16),
        base=np.array(base, np.int32), inc=np.array(inc, np.int16),
        result=np.array(result, np.int8),
    )


def maia_protocol_mask(gs: GameStore, skip: int = 10, pressure: int = 30) -> np.ndarray:
    """Per-ply bool mask of the Maia-3 / ALLIE evaluation protocol: drop each game's first
    ``skip`` plies and every ply from the first one whose mover has < ``pressure`` s left.
    On the ALLIE test set this selects exactly the 884,049 positions reported in the paper."""
    keep = np.zeros(len(gs.moves), bool)
    for g in range(gs.n_games):
        a, b = gs.offsets[g], gs.offsets[g + 1]
        low = np.flatnonzero(gs.clock[a:b] < pressure)
        end = a + (low[0] if len(low) else b - a)
        keep[a + skip:end] = True
    return keep
