"""Correctness tests: python -m pytest -q tests  (needs python-chess; AVX-512 for the kernel tests)."""
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

chess = pytest.importorskip("chess")

from cloverb import core  # noqa: E402
from cloverb.gamestore import decode_uci, encode_uci  # noqa: E402
from cloverb.model import Cloverb, CloverbConfig  # noqa: E402
import cloverb.model as M  # noqa: E402

PERSP = lambda s, black: s ^ 56 if black else s  # noqa: E731


def random_game(rng, plies=80):
    b, moves = chess.Board(), []
    while len(moves) < plies and not b.is_game_over():
        m = rng.choice(list(b.legal_moves))
        moves.append(m.uci()); b.push(m)
    return moves


def vocab(m: chess.Move, white: bool) -> int:
    f, t = m.from_square, m.to_square
    if not white:
        f, t = f ^ 56, t ^ 56
    if m.promotion:
        return 4096 + ((f & 7) * 8 + (t & 7)) * 4 + {5: 0, 4: 1, 3: 2, 2: 3}[m.promotion]
    return f * 64 + t


def test_legal_moves_and_encoding_match_python_chess():
    rng = random.Random(0)
    for _ in range(200):
        g = random_game(rng, rng.randint(0, 120))
        b = chess.Board()
        for u in g:
            b.push_uci(u)
        enc = core.encode_position(None, np.array([encode_uci(u) for u in g], np.uint16))
        board, meta, hist, legal, lm = enc
        assert sorted(decode_uci(int(x)) for x in lm) == sorted(m.uci() for m in b.legal_moves)
        bits = np.flatnonzero(np.unpackbits(legal, bitorder="little"))
        assert sorted(bits.tolist()) == sorted(vocab(m, b.turn) for m in b.legal_moves)
        black = not b.turn
        for s in range(64):
            p = b.piece_at(PERSP(s, black))
            assert board[s] == (0 if p is None else (p.piece_type if p.color == b.turn else 6 + p.piece_type))
        assert meta[5] == b.halfmove_clock and meta[6] == b.fullmove_number and meta[7] == b.is_check()


def test_parse_games_with_clocks():
    txt = "1. e4 { [%clk 0:03:00] } 1... c5 { [%clk 0:02:59] } 2. Nf3 { [%clk 0:02:58] } 2... d6 { [%clk 0:02:57] } 1-0"
    mv, clk, nply = core.parse_games([txt])
    assert nply[0] == 4 and [decode_uci(int(m)) for m in mv] == ["e2e4", "c7c5", "g1f3", "d7d6"]
    assert clk.tolist() == [180, 179, 178, 177]


@pytest.mark.skipif(not M.FUSED_ATTN, reason="needs AVX-512")
@pytest.mark.parametrize("postln", [False, True])
def test_fused_attention_matches_pytorch(postln):
    torch.manual_seed(0)
    cfg = CloverbConfig(d=256, layers=2, mlp_mult=2, n_templates=4, geo_init=True, postln=postln, cap_type=True)
    m = Cloverb(cfg)
    with torch.no_grad():
        for b in m.blocks:
            b.dyn.weight.normal_(0, 0.05); b.proj.weight.normal_(0, 0.02); b.fc2.weight.normal_(0, 0.02)
    B = 4
    board = torch.randint(0, 13, (B, 64), dtype=torch.int8)
    meta = torch.zeros(B, 8, dtype=torch.int16); meta[:, 4] = -1; meta[:, 6] = 20
    hist = torch.randint(0, 64, (B, 64, 4), dtype=torch.int8); hist[..., 2] %= 6; hist[..., 3] %= 6
    f = lambda v: torch.full((B,), float(v))  # noqa: E731
    outs = []
    for fused in (True, False):
        M.FUSED_ATTN = fused
        lo = m(board, meta, hist, f(1500), f(180), f(2), f(100))[0]
        lo.float().logsumexp(1).sum().backward()
        outs.append((lo.detach(), [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in m.parameters()]))
        m.zero_grad()
    M.FUSED_ATTN = True
    assert (outs[0][0] - outs[1][0]).abs().max() < 1e-4
    for a, b in zip(outs[0][1], outs[1][1]):
        assert (a - b).abs().max() <= 1e-4 * (b.abs().max() + 1e-6)


@pytest.mark.skipif(not M.FUSED_ATTN, reason="needs AVX-512")
def test_c_engine_matches_pytorch(tmp_path):
    torch.manual_seed(1)
    cfg = CloverbConfig(d=256, layers=2, heads=8, mlp_mult=3, n_templates=0, geo_init=True, postln=True, cap_type=True)
    m = Cloverb(cfg)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    ck = tmp_path / "m.pt"
    torch.save(dict(ema=m.state_dict(), cfg=cfg.to_dict(), step=0), ck)
    subprocess.check_call([sys.executable, str(ROOT / "tools/export_engine.py"), str(ck), str(tmp_path / "m.bin")])
    exe = tmp_path / "engine"
    subprocess.check_call(["gcc", "-O3", "-march=native", "-o", str(exe), str(ROOT / "csrc/cloverb_engine.c"), "-lm"])
    M.QK_EPS = 2 ** -7
    try:
        moves = "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6".split()
        b, meta, hist, legal, lm = core.encode_position(None, np.array([encode_uci(u) for u in moves], np.uint16))
        t = lambda a: torch.from_numpy(a)[None]  # noqa: E731
        m.eval()
        with torch.no_grad():
            lo = m(t(b), t(meta), t(hist), torch.tensor([1700.]), torch.tensor([300.]), torch.tensor([3.]), torch.tensor([250.]))[0][0]
        out = subprocess.run([str(exe), str(tmp_path / "m.bin"), "dump", "1700", "300", "3", "250"] + moves,
                             capture_output=True, check=True).stdout
        cl = np.frombuffer(out, np.float32)[:4352]
        mask = np.unpackbits(legal, bitorder="little").astype(bool)
        sm = lambda x: np.exp(x - x.max()) / np.exp(x - x.max()).sum()  # noqa: E731
        assert np.abs(sm(lo.numpy()[mask]) - sm(cl[mask])).max() < 5e-3
    finally:
        M.QK_EPS = None
