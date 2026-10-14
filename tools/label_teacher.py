"""Label a pool of training positions with a Maia-3 teacher (top-k legal-move probabilities).

The pool is drawn with the training sampler; for each position we store (game, ply) and the teacher's
top-k moves (vocabulary index, probability). Training on the pool for several epochs amortises the
teacher's cost (see cloverb.train --teacher-cache).

usage: python tools/label_teacher.py --stores S1.npz ... --out pool_000.npz --n 2000000 [--teacher maia3-23m] [--device cuda]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloverb.baseline import maia_inputs  # noqa: E402
from cloverb.data import Prefetcher, Sampler, SamplerConfig, concat_stores  # noqa: E402
from cloverb.teacher import FastMaia3, teacher_logprobs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stores", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2_000_000)
    ap.add_argument("--teacher", default="maia3-23m")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--elo-alpha", type=float, default=0.3)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.set_num_threads(4)
    gs = concat_stores(a.stores)
    sampler = Sampler(gs, SamplerConfig(elo_alpha=a.elo_alpha, seed=a.seed))
    teacher = torch.compile(FastMaia3(a.teacher).to(dev))
    G = np.empty(a.n, np.int64); P = np.empty(a.n, np.int32)
    IDX = np.empty((a.n, a.k), np.int16); PR = np.empty((a.n, a.k), np.float16)
    t0 = time.time()
    def extra(gs_, g, ply):
        xm, se, oe = maia_inputs(gs_, g, ply)
        return dict(maia_x=xm, se=se, oe=oe)
    pf = Prefetcher(gs, sampler, a.batch, 1, extra=extra, pin=dev.type == "cuda")
    for s in range(0, a.n, a.batch):
        B = min(a.batch, a.n - s)
        x = next(pf)
        g, ply, xm, se, oe = x["g"][:B], x["ply"][:B], x["maia_x"][:B], x["se"][:B], x["oe"][:B]
        x["legal"] = x["legal"][:B]
        for k in ("base", "inc", "clock"):
            x[k] = x[k][:B]
        tm = torch.stack([x["base"], x["inc"], x["clock"]], 1)
        d = lambda t: t.to(dev, non_blocking=True)  # noqa: E731
        lp = teacher_logprobs(teacher, d(xm), d(se), d(oe), d(x["legal"]), time=d(tm))
        p, i = (v.cpu() for v in lp.exp().topk(a.k, dim=1))
        G[s:s + B], P[s:s + B] = g, ply
        IDX[s:s + B] = i.numpy().astype(np.int16)
        PR[s:s + B] = p.numpy().astype(np.float16)
        if (s // a.batch) % 200 == 0:
            print(f"{s + B}/{a.n} {(s + B) / (time.time() - t0):.0f} pos/s", flush=True)
    pf.close()
    np.savez(a.out, game=G, ply=P, idx=IDX, prob=PR, stores=np.array(a.stores))
    print(f"done {a.n} in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
