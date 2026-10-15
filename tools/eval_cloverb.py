"""Score a Cloverb checkpoint on a game store subset; same output format as tools/bench_maia3.py.

usage: python tools/eval_cloverb.py --ckpt runs/X/ema.pt --store S.npz --subset SUB.npy --out R.npz
       [--no-clock]  (replace clocks with a neutral value, for ablation)
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloverb.data import MODEL_KEYS, make_inputs  # noqa: E402
from cloverb.gamestore import GameStore  # noqa: E402
from cloverb.infer import load  # noqa: E402
from cloverb.model import TIME_BINS, time_bucket  # noqa: E402


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
    ap.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp32"],
                    help="trunk precision; auto = bf16 only on CPUs with AMX (elsewhere bf16 GEMMs are emulated)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    bf16 = a.precision == "bf16" or (a.precision == "auto" and "amx_bf16" in Path("/proc/cpuinfo").read_text())
    if not bf16:  # same function as under bf16 autocast: keep the q/k RMSNorm eps of bf16 training
        import cloverb.model as cm
        cm.QK_EPS = torch.finfo(torch.bfloat16).eps
    model = load(a.ckpt)
    H = model.cfg.hist
    gs = GameStore.load(a.store)
    idx = np.load(a.subset)
    g = np.searchsorted(gs.offsets, idx, side="right") - 1
    ply = idx - gs.offsets[g]
    n = len(idx)
    ce = np.zeros(n, np.float32); rank = np.zeros(n, np.int16)
    # think time: predicted median (s), true time spent (-1 = unknown), bucket CE; value: WDL CE / argmax hit
    tmed = np.zeros(n, np.float32); tspent = np.zeros(n, np.float32); tce = np.zeros(n, np.float32)
    tbase = np.zeros(n, np.float32); vce = np.zeros(n, np.float32); vhit = np.zeros(n, bool)
    centers = torch.expm1(torch.arange(TIME_BINS) * math.log1p(1200) / (TIME_BINS - 1))
    t0 = time.perf_counter()
    with torch.inference_mode():
        for s in range(0, n, a.batch):
            x = make_inputs(gs, g[s:s + a.batch], ply[s:s + a.batch], H)
            if a.no_clock:  # break the clock/position link: shuffle the fraction of base time left
                frac = (x["clock"] / x["base"].clamp(min=1))[torch.randperm(len(x["clock"]))]
                x["clock"] = frac * x["base"]
            if a.hist_limit is not None:
                x["hist"][:, a.hist_limit:] = -1
            if a.elo_override is not None:
                x["elo"] = torch.full_like(x["elo"], a.elo_override)
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
                lo, v, tt = model(*(x[k] for k in MODEL_KEYS))
            lo = lo.float().masked_fill(~x["legal"], float("-inf"))
            t = x["target"]; tl = lo.gather(1, t[:, None])
            sl = slice(s, s + a.batch)
            ce[sl] = (torch.logsumexp(lo, 1) - tl[:, 0]).numpy()
            rank[sl] = (lo > tl).sum(1).numpy()
            v = v.float(); vce[sl] = F.cross_entropy(v, x["wdl"], reduction="none").numpy()
            vhit[sl] = (v.argmax(1) == x["wdl"]).numpy()
            if tt is not None:
                tt = tt.float(); cdf = tt.softmax(1).cumsum(1)
                tmed[sl] = centers[(cdf < 0.5).sum(1).clamp(max=TIME_BINS - 1)].numpy()
                tce[sl] = F.cross_entropy(tt, time_bucket(x["spent"].clamp(min=0)), reduction="none").numpy()
            tspent[sl] = x["spent"].numpy()
            tbase[sl] = (x["clock"] / 40 + x["inc"]).numpy()  # rule-of-thumb baseline: 1/40 of the clock + increment
    dt = time.perf_counter() - t0
    top1, top3 = float(np.mean(rank == 0)), float(np.mean(rank < 3))
    se = lambda p: 1.96 * np.sqrt(p * (1 - p) / n)  # noqa: E731
    info = dict(variant=Path(a.ckpt).parent.name, label=a.label, hist_limit=a.hist_limit, ckpt=a.ckpt, positions=int(n), top1=top1, top1_ci=se(top1),
                top3=top3, top3_ci=se(top3), ce=float(ce.mean()), seconds=dt, pos_per_s=n / dt,
                store=a.store, subset=a.subset, no_clock=a.no_clock,
                value_ce=float(vce.mean()), value_acc=float(vhit.mean()), precision="bf16" if bf16 else "fp32")
    k = tspent >= 0
    if k.any() and tmed.any():
        lt, lp, lb = np.log1p(tspent[k]), np.log1p(tmed[k]), np.log1p(tbase[k])
        info.update(time_positions=int(k.sum()), time_ce=float(tce[k].mean()),
                    time_mae_log=float(np.abs(lp - lt).mean()),                 # |log(1+pred) - log(1+true)|
                    time_mae_log_const=float(np.abs(np.median(lt) - lt).mean()),  # best constant (oracle median)
                    time_mae_log_rule=float(np.abs(lb - lt).mean()),            # clock/40 + increment
                    time_corr_log=float(np.corrcoef(lp, lt)[0, 1]),
                    time_within_2x=float(np.mean(np.abs(np.log((tmed[k] + 0.5) / (tspent[k] + 0.5))) <= np.log(2))))
    np.savez(a.out, idx=idx, ce=ce, rank=rank, tmed=tmed, tspent=tspent)
    Path(a.out).with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
