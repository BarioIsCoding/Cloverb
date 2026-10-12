"""ctypes bindings for csrc/cloverb_core.c (compiled on first import if missing or stale)."""
from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "csrc" / "cloverb_core.c"
LIB = HERE / "libcloverb.so"
VOCAB = 4352  # 64*64 from-to + 8*8*4 promotions, Maia-3 compatible


def _load():
    if SRC.exists() and (not LIB.exists() or LIB.stat().st_mtime < SRC.stat().st_mtime):
        subprocess.check_call(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", str(LIB), str(SRC)])
    lib = ctypes.CDLL(str(LIB))
    P = np.ctypeslib.ndpointer
    lib.cb_parse_games.restype = ctypes.c_int64
    lib.cb_parse_games.argtypes = [ctypes.c_char_p, P(np.int64), ctypes.c_int, P(np.uint16), P(np.int32),
                                   P(np.int32), ctypes.c_int64, ctypes.c_int]
    lib.cb_build_batch.restype = ctypes.c_int
    lib.cb_build_batch.argtypes = [P(np.uint16), P(np.int64), P(np.int64), P(np.int32), ctypes.c_int, ctypes.c_int,
                                   P(np.int8), P(np.int16), P(np.int8), ctypes.c_void_p, ctypes.c_void_p]
    lib.cb_build_maia.restype = ctypes.c_int
    lib.cb_build_maia.argtypes = [P(np.uint16), P(np.int64), P(np.int64), P(np.int32), ctypes.c_int, ctypes.c_int,
                                  P(np.uint64)]
    lib.cb_legal_after.restype = ctypes.c_int
    lib.cb_legal_after.argtypes = [P(np.uint16), ctypes.c_int, P(np.uint16)]
    return lib


lib = _load()


def parse_games(texts: list[str], max_ply: int = 1000):
    """SAN movetexts -> (moves uint16[P], clk_after int32[P], nply int32[G]); nply=-1 marks a failed game."""
    enc = [t.encode() for t in texts]
    offs = np.zeros(len(enc) + 1, np.int64)
    np.cumsum([len(e) for e in enc], out=offs[1:])
    buf = b"".join(enc)
    cap = max(16, len(buf) // 3)
    moves = np.empty(cap, np.uint16)
    clk = np.empty(cap, np.int32)
    nply = np.empty(len(enc), np.int32)
    w = lib.cb_parse_games(buf, offs, len(enc), moves, clk, nply, cap, max_ply)
    if w < 0:
        raise RuntimeError("parse buffer overflow")
    return moves[:w].copy(), clk[:w].copy(), nply


class Batch:
    __slots__ = ("board", "meta", "hist", "legal", "target")


def build_batch(moves: np.ndarray, offsets: np.ndarray, gidx: np.ndarray, ply: np.ndarray, H: int = 64,
                legal: bool = True, target: bool = True) -> Batch:
    n = len(gidx)
    b = Batch()
    b.board = np.empty((n, 64), np.int8)
    b.meta = np.empty((n, 8), np.int16)
    b.hist = np.empty((n, H, 4), np.int8)
    b.legal = np.empty((n, VOCAB // 8), np.uint8) if legal else None
    b.target = np.empty(n, np.int16) if target else None
    rc = lib.cb_build_batch(moves, offsets, np.ascontiguousarray(gidx, np.int64), np.ascontiguousarray(ply, np.int32),
                            n, H, b.board, b.meta, b.hist,
                            b.legal.ctypes.data if legal else None, b.target.ctypes.data if target else None)
    if rc:
        raise ValueError(f"illegal stored game at sample {rc - 1}")
    return b


def build_maia(moves, offsets, gidx, ply, H: int = 8) -> np.ndarray:
    out = np.empty((len(gidx), H, 12), np.uint64)
    lib.cb_build_maia(moves, offsets, np.ascontiguousarray(gidx, np.int64), np.ascontiguousarray(ply, np.int32),
                      len(gidx), H, out)
    return out


def legal_after(moves: np.ndarray) -> np.ndarray:
    out = np.empty(256, np.uint16)
    n = lib.cb_legal_after(np.ascontiguousarray(moves, np.uint16), len(moves), out)
    return out[:n].copy()
