"""Is an ensemble of Maia-3 checkpoints a better teacher than the best single one?

Scores each Maia-3 variant and probability / log-probability averages of them on the training-time
evaluation sample (ALLIE, Maia protocol, 10k positions), optionally together with a Cloverb checkpoint.
usage: python tools/ensemble_teacher.py [--cloverb CKPT] [--out results/maia3/ensemble10k.json]
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloverb.baseline import maia_inputs  # noqa: E402
from cloverb.data import MODEL_KEYS, make_inputs  # noqa: E402
from cloverb.gamestore import GameStore  # noqa: E402
from cloverb.teacher import FastMaia3, teacher_logprobs  # noqa: E402


def score(lp, t):
    tl = lp.gather(1, t[:, None])
    rank = (lp > tl).sum(1)
    return dict(top1=float((rank == 0).float().mean()), top3=float((rank < 3).float().mean()), ce=float(-tl.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="/home/user/data/bench/allie2022.npz")
    ap.add_argument("--idx", default="/home/user/data/bench/allie_sub10k.npy")
    ap.add_argument("--models", default="maia3-79m,maia3-23m,maia3-23m-ponder,maia3-5m")
    ap.add_argument("--cloverb", default=None)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--out", default="results/maia3/ensemble10k.json")
    a = ap.parse_args()
    gs = GameStore.load(a.store)
    idx = np.load(a.idx)
    g = np.searchsorted(gs.offsets, idx, side="right") - 1
    ply = idx - gs.offsets[g]
    names = a.models.split(",") + (["cloverb"] if a.cloverb else [])
    lps = {n: [] for n in names}
    tgt = []
    nets = {n: FastMaia3(n).eval() for n in a.models.split(",")}
    if a.cloverb:
        from cloverb.infer import load
        nets["cloverb"] = load(a.cloverb).eval()
    for s in range(0, len(idx), a.batch):
        sl = slice(s, s + a.batch)
        x = make_inputs(gs, g[sl], ply[sl], 64)
        mx, se, oe = maia_inputs(gs, g[sl], ply[sl])
        tm = torch.stack([x["base"], x["inc"], x["clock"]], 1)
        tgt.append(x["target"])
        for n, m in nets.items():
            if n == "cloverb":
                with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
                    lo = m(*(x[k] for k in MODEL_KEYS))[0]
                lp = torch.log_softmax(lo.float().masked_fill(~x["legal"], float("-inf")), 1)
            else:
                lp = teacher_logprobs(m, mx, se, oe, x["legal"], time=tm if "ponder" in n else None)
            lps[n].append(lp.half())
        print(f"{s + a.batch}/{len(idx)}", flush=True)
    t = torch.cat(tgt)
    lps = {n: torch.cat(v).float() for n, v in lps.items()}
    res = {n: score(lp, t) for n, lp in lps.items()}
    for k in range(2, len(names) + 1):
        for combo in itertools.combinations(names, k):
            st = torch.stack([lps[n] for n in combo])
            res["+".join(combo) + " (prob mean)"] = score(torch.logsumexp(st, 0) - np.log(k), t)
            gm = st.mean(0)  # -inf stays -inf on illegal moves
            res["+".join(combo) + " (logit mean)"] = score(torch.log_softmax(gm.masked_fill(torch.isnan(gm), float("-inf")), 1), t)
    for n, r in sorted(res.items(), key=lambda kv: -kv[1]["top1"]):
        print(f"{100 * r['top1']:6.2f} {100 * r['top3']:6.2f} {r['ce']:.4f}  {n}")
    Path(a.out).write_text(json.dumps(dict(positions=len(idx), results=res), indent=2) + "\n")


if __name__ == "__main__":
    main()
