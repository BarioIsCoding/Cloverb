"""Score a Cloverb checkpoint on a game store subset; same output format as tools/bench_maia3.py.

usage: python tools/eval_cloverb.py --ckpt runs/X/ema.pt --store S.npz --subset SUB.npy --out R.npz
       [--no-clock]  (replace clocks with a neutral value, for ablation)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloverb.data import MODEL_KEYS, make_inputs  # noqa: E402
from cloverb.gamestore import GameStore  # noqa: E402
from cloverb.infer import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-clock", action="store_true")
    ap.add_argument("--elo-override", type=float, default=None)
    ap.add_argument("--hist-limit", type=int, default=None, help="hide plies older than this (ablation)")
    ap.add_argument("--label", default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    model = load(a.ckpt)
    H = model.cfg.hist
    gs = GameStore.load(a.store)
    idx = np.load(a.subset)
    g = np.searchsorted(gs.offsets, idx, side="right") - 1
    ply = idx - gs.offsets[g]
    ce = np.zeros(len(idx), np.float32); rank = np.zeros(len(idx), np.int16)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for s in range(0, len(idx), a.batch):
            x = make_inputs(gs, g[s:s + a.batch], ply[s:s + a.batch], H)
            if a.no_clock:  # break the clock/position link: shuffle the fraction of base time left
                frac = (x["clock"] / x["base"].clamp(min=1))[torch.randperm(len(x["clock"]))]
                x["clock"] = frac * x["base"]
            if a.hist_limit is not None:
                x["hist"][:, a.hist_limit:] = -1
            if a.elo_override is not None:
                x["elo"] = torch.full_like(x["elo"], a.elo_override)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                lo = model(*(x[k] for k in MODEL_KEYS))[0]
            lo = lo.float().masked_fill(~x["legal"], float("-inf"))
            t = x["target"]; tl = lo.gather(1, t[:, None])
            ce[s:s + a.batch] = (torch.logsumexp(lo, 1) - tl[:, 0]).numpy()
            rank[s:s + a.batch] = (lo > tl).sum(1).numpy()
    dt = time.perf_counter() - t0
    top1, top3 = float(np.mean(rank == 0)), float(np.mean(rank < 3))
    se = lambda p: 1.96 * np.sqrt(p * (1 - p) / len(idx))
    info = dict(variant=Path(a.ckpt).parent.name, label=a.label, hist_limit=a.hist_limit, ckpt=a.ckpt, positions=int(len(idx)), top1=top1, top1_ci=se(top1),
                top3=top3, top3_ci=se(top3), ce=float(ce.mean()), seconds=dt, pos_per_s=len(idx) / dt,
                store=a.store, subset=a.subset, no_clock=a.no_clock)
    np.savez(a.out, idx=idx, ce=ce, rank=rank)
    Path(a.out).with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
