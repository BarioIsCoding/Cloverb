"""Teacher accuracy by time control on held-out games of the training stores.

Picks whole games from the training stores per base-time bucket (seeded; these games are then kept out of
any new teacher pool), writes their plies as a subset file, and scores Maia-3 teachers on them.
Used to decide where the clock-aware teacher can be trusted, without looking at the test sets.

usage: python tools/teacher_by_tc.py --stores S... --subset-out trainval_tc.npz --out results/teacher_by_tc.json
       [--teachers maia3-23m-ponder maia3-23m] [--games 120]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cloverb.baseline import maia_inputs  # noqa: E402
from cloverb.data import concat_stores, make_inputs  # noqa: E402
from cloverb.teacher import FastMaia3, teacher_logprobs  # noqa: E402

# base-time buckets (seconds): blitz control, 10 min, 15-20 min, 25 min and longer
BUCKETS = [("blitz 3-5 min", 180, 301), ("rapid 10 min", 600, 601), ("rapid 15-20 min", 900, 1500),
           ("classical 25+ min", 1500, 1 << 30)]


def bucket_of(base):
    b = np.full(len(base), -1)
    for i, (_, lo, hi) in enumerate(BUCKETS):
        b[(base >= lo) & (base < hi)] = i
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stores", nargs="+", required=True)
    ap.add_argument("--subset-out", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--teachers", nargs="*", default=["maia3-23m-ponder", "maia3-23m"])
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    torch.set_num_threads(4)
    gs = concat_stores(a.stores)
    rng = np.random.default_rng(a.seed)
    lens = np.diff(gs.offsets)
    gb = bucket_of(gs.base)
    if Path(a.subset_out).exists():
        z = np.load(a.subset_out)
        games = z["games"]
    else:
        ok = (lens >= 20) & (gs.welo > 0) & (gs.belo > 0)
        games = np.sort(np.concatenate([rng.choice(np.flatnonzero(ok & (gb == i)), a.games, replace=False)
                                        for i in range(len(BUCKETS))]))
    g = np.repeat(games, lens[games])
    ply = np.concatenate([np.arange(n) for n in lens[games]])
    idx = gs.offsets[g] + ply
    np.savez(a.subset_out, games=games, idx=idx)
    pb = gb[g]
    res = {"games_per_bucket": a.games, "positions": {BUCKETS[i][0]: int((pb == i).sum()) for i in range(len(BUCKETS))},
           "teachers": {}}
    for name in a.teachers:
        teacher = torch.compile(FastMaia3(name))
        hit = np.zeros(len(g), bool)
        t0 = time.time()
        for s in range(0, len(g), a.batch):
            gg, pp = g[s:s + a.batch], ply[s:s + a.batch]
            x = make_inputs(gs, gg, pp, H=1)
            xm, se, oe = maia_inputs(gs, gg, pp)
            tm = torch.stack([x["base"], x["inc"], x["clock"]], 1)
            lp = teacher_logprobs(teacher, xm, se, oe, x["legal"], time=tm)
            hit[s:s + a.batch] = (lp.argmax(1) == x["target"]).numpy()
        res["teachers"][name] = {BUCKETS[i][0]: 100 * float(hit[pb == i].mean()) for i in range(len(BUCKETS))}
        print(name, res["teachers"][name], f"{len(g) / (time.time() - t0):.0f} pos/s", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
