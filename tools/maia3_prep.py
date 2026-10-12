"""Precompute everything the Maia-3 scorer needs per ply of a game store, once.

Output npz (per ply of the store):
  planes  uint64[P,12]  Maia-3 tokenization as bitboards (board mirrored to the mover, own P..K, opp p..k)
  target  int16[P]      Maia-3 vocabulary index of the played move
  legal   int16[L]      ragged legal-move vocabulary indices, legal_off int64[P+1]
  check   bool[P]       mover is in check
Usage: python tools/maia3_prep.py STORE.npz OUT.npz
"""
import sys
from multiprocessing import Pool

import chess
import numpy as np

sys.path.insert(0, __file__.rsplit("/tools/", 1)[0])
from cloverb.gamestore import GameStore, decode_uci  # noqa: E402

PROMO_IDX = {chess.QUEEN: 0, chess.ROOK: 1, chess.BISHOP: 2, chess.KNIGHT: 3}


def maia_index(move: chess.Move, white: bool) -> int:
    f, t = move.from_square, move.to_square
    if not white:
        f, t = f ^ 56, t ^ 56
    if move.promotion:
        return 4096 + ((f & 7) * 8 + (t & 7)) * 4 + PROMO_IDX[move.promotion]
    return f * 64 + t


def planes(board: chess.Board) -> list:
    white = board.turn
    own, opp = (chess.WHITE, chess.BLACK) if white else (chess.BLACK, chess.WHITE)
    out = []
    for color in (own, opp):
        for pt in range(1, 7):
            bb = board.pieces_mask(pt, color)
            out.append(bb if white else chess.flip_vertical(bb))
    return out


def work(args):
    moves, offsets = args
    P = len(moves)
    pl = np.zeros((P, 12), np.uint64)
    tgt = np.zeros(P, np.int16)
    chk = np.zeros(P, bool)
    legal, loff = [], [0]
    for g in range(len(offsets) - 1):
        b = chess.Board()
        for k in range(offsets[g], offsets[g + 1]):
            pl[k] = planes(b)
            chk[k] = b.is_check()
            lm = [maia_index(m, b.turn) for m in b.legal_moves]
            legal.extend(lm); loff.append(loff[-1] + len(lm))
            mv = chess.Move.from_uci(decode_uci(int(moves[k])))
            tgt[k] = maia_index(mv, b.turn)
            b.push(mv)
    return pl, tgt, chk, np.array(legal, np.int16), np.array(loff, np.int64)


def main():
    gs = GameStore.load(sys.argv[1])
    n = gs.n_games
    cuts = np.linspace(0, n, 33).astype(int)
    jobs = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        o = gs.offsets[a:b + 1]
        jobs.append((gs.moves[o[0]:o[-1]], (o - o[0])))
    with Pool(4) as pool:
        res = pool.map(work, jobs)
    pl = np.concatenate([r[0] for r in res]); tgt = np.concatenate([r[1] for r in res])
    chk = np.concatenate([r[2] for r in res])
    legal = np.concatenate([r[3] for r in res])
    loff = [np.zeros(1, np.int64)]
    base = 0
    for r in res:
        loff.append(r[4][1:] + base); base += r[4][-1]
    loff = np.concatenate(loff)
    assert len(pl) == len(gs.moves) and loff[-1] == len(legal)
    np.savez(sys.argv[2], planes=pl, target=tgt, check=chk, legal=legal, legal_off=loff)
    print(f"{len(pl)} plies, {len(legal)} legal moves ({len(legal)/len(pl):.1f}/ply)")


if __name__ == "__main__":
    main()
