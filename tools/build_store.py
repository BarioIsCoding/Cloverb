"""Lichess parquet shard(s) -> Cloverb game store (see cloverb/gamestore.py).

Keeps games with clock annotations, both ratings, a parseable movetext, >= 2 plies,
termination Normal / Time forfeit, and an estimated duration (base + 40*inc) >= 30 s
(i.e. everything from bullet to classical; UltraBullet is dropped).

usage: python tools/build_store.py OUT.npz SHARD.parquet [SHARD.parquet ...] [--workers 4]
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloverb import core  # noqa: E402
from cloverb.gamestore import GameStore  # noqa: E402

COLS = ["WhiteElo", "BlackElo", "TimeControl", "Termination", "Result", "movetext"]
RESULT = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}


def work(job):
    path, rgs = job
    pf = pq.ParquetFile(path)
    t = pf.read_row_groups(rgs, columns=COLS).to_pydict()
    keep_txt, meta = [], []
    for we, be, tc, term, res, mt in zip(*(t[c] for c in COLS)):
        if we is None or be is None or tc is None or "+" not in tc or term not in ("Normal", "Time forfeit"):
            continue
        if res not in RESULT or "%clk" not in mt:
            continue
        base, inc = tc.split("+")
        base, inc = int(base), int(inc)
        if base + 40 * inc < 30:
            continue
        keep_txt.append(mt)
        meta.append((we, be, base, inc, RESULT[res]))
    mv, clk_after, nply = core.parse_games(keep_txt)
    # assemble per-game arrays, computing clock before each ply and time spent
    out_m, out_c, out_s, lens, gm = [], [], [], [], []
    off = 0
    for (we, be, base, inc, res), n in zip(meta, nply):
        if n < 0:
            continue
        m, ca = mv[off:off + n], clk_after[off:off + n]
        off += n
        if n < 2 or (ca < 0).any():
            continue
        before = np.empty(n, np.int64)
        before[:2] = base
        before[2:] = ca[:-2]
        spent = np.clip(before - ca + inc, 0, 65535)
        out_m.append(m); out_c.append(np.clip(before, 0, 65535)); out_s.append(spent)
        lens.append(n); gm.append((we, be, base, inc, res))
    gm = np.array(gm, np.int64).reshape(-1, 5)
    return (np.concatenate(out_m) if out_m else np.zeros(0, np.uint16),
            np.concatenate(out_c).astype(np.uint16) if out_c else np.zeros(0, np.uint16),
            np.concatenate(out_s).astype(np.uint16) if out_s else np.zeros(0, np.uint16),
            np.array(lens, np.int64), gm, len(t["movetext"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rg-per-job", type=int, default=40)
    a = ap.parse_args()
    jobs = []
    for s in a.shards:
        n = pq.ParquetFile(s).metadata.num_row_groups
        jobs += [(s, list(range(i, min(n, i + a.rg_per_job)))) for i in range(0, n, a.rg_per_job)]
    with Pool(a.workers) as pool:
        res = pool.map(work, jobs, chunksize=1)
    lens = np.concatenate([r[3] for r in res])
    gm = np.concatenate([r[4] for r in res])
    offsets = np.zeros(len(lens) + 1, np.int64)
    np.cumsum(lens, out=offsets[1:])
    gs = GameStore(moves=np.concatenate([r[0] for r in res]), clock=np.concatenate([r[1] for r in res]),
                   spent=np.concatenate([r[2] for r in res]), offsets=offsets,
                   welo=gm[:, 0].astype(np.int16), belo=gm[:, 1].astype(np.int16), base=gm[:, 2].astype(np.int32),
                   inc=gm[:, 3].astype(np.int16), result=gm[:, 4].astype(np.int8))
    gs.save(a.out)
    seen = sum(r[5] for r in res)
    print(f"{seen} games read, {gs.n_games} kept, {len(gs.moves)} plies -> {a.out}")


if __name__ == "__main__":
    main()
