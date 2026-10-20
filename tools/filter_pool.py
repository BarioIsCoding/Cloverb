"""Drop entries from teacher pools: positions of games with a long base time and/or of given held-out games.

usage: python tools/filter_pool.py --stores S... --pools pool_*.npz --out-dir DIR [--max-base 600] [--exclude games.npz]
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stores", nargs="+", required=True, help="the stores the pools index (merged directory or .npz files)")
    ap.add_argument("--pools", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-base", type=int, default=None, help="keep only games with base time <= this (seconds)")
    ap.add_argument("--exclude", default=None, help="npz with a 'games' array of game indices to drop")
    a = ap.parse_args()
    base = np.concatenate([np.load(s + "/base.npy") if Path(s).is_dir() else np.load(s)["base"] for s in a.stores])
    drop = np.zeros(len(base), bool)
    if a.max_base is not None:
        drop |= base > a.max_base
    if a.exclude:
        drop[np.load(a.exclude)["games"]] = True
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    for p in a.pools:
        z = np.load(p)
        keep = ~drop[z["game"]]
        np.savez(Path(a.out_dir) / Path(p).name, **{k: (z[k][keep] if z[k].shape[:1] == z["game"].shape else z[k])
                                                   for k in z.files})
        print(f"{p}: kept {keep.sum()} of {len(keep)}", flush=True)


if __name__ == "__main__":
    main()
