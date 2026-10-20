"""Compare the standalone C engine with the PyTorch model on the trained weights.

For N random positions of a game store, runs `ENGINE weights.bin dump ELO BASE INC CLOCK moves...` and
the PyTorch forward pass in fp32 (q/k RMSNorm eps of the bf16 build), and reports the largest
legal-move probability difference, KL divergence and top-move agreement.

usage: python tools/check_engine.py --ckpt checkpoints/cloverb.pt --bin checkpoints/cloverb.bin
       --engine ./cloverb-engine [--engine ./cloverb-engine-fp32] --store S.npz --n 500 --out R.json
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cloverb import core  # noqa: E402
from cloverb import model as M  # noqa: E402
from cloverb.gamestore import GameStore, decode_uci  # noqa: E402
from cloverb.infer import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--engine", action="append", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    M.QK_EPS = torch.finfo(torch.bfloat16).eps
    m = load(a.ckpt)
    H = m.cfg.hist
    gs = GameStore.load(a.store)
    rng = np.random.default_rng(a.seed)
    res = {e: dict(maxdp=[], kl=[], top=[]) for e in a.engine}
    n = 0
    while n < a.n:
        g = int(rng.integers(len(gs.offsets) - 1))
        lo, hi = int(gs.offsets[g]), int(gs.offsets[g + 1])
        if hi - lo < 2:
            continue
        ply = int(rng.integers(hi - lo))
        k = lo + ply
        mv = np.asarray(gs.moves[lo:k], np.uint16)
        enc = core.encode_position(None, mv, H)
        if enc is None or len(enc[4]) < 2:
            continue
        b, meta, hist, legal, _ = enc
        elo = float(gs.welo[g] if ply % 2 == 0 else gs.belo[g])
        base, inc, clk = float(gs.base[g]), float(gs.inc[g]), float(gs.clock[k])
        t = lambda x: torch.from_numpy(x)[None]  # noqa: E731
        with torch.no_grad():
            lp = m(t(b), t(meta), t(hist), torch.tensor([elo]), torch.tensor([base]), torch.tensor([inc]),
                   torch.tensor([clk]))[0][0].numpy()
        mask = np.unpackbits(legal, bitorder="little").astype(bool)[:lp.shape[0]]
        ref = lp[mask] - lp[mask].max()
        p = np.exp(ref) / np.exp(ref).sum()
        args = [str(elo), str(base), str(inc), str(clk)] + [decode_uci(int(x)) for x in mv]
        for e in a.engine:
            out = subprocess.run([e, a.bin, "dump", *args], capture_output=True, check=True).stdout
            cl = np.frombuffer(out, np.float32)[:4352][mask]
            cl = cl - cl.max()
            q = np.exp(cl) / np.exp(cl).sum()
            res[e]["maxdp"].append(float(np.abs(p - q).max()))
            res[e]["kl"].append(float(np.sum(p * (np.log(p + 1e-30) - np.log(q + 1e-30)))))
            res[e]["top"].append(bool(p.argmax() == q.argmax()))
        n += 1
    info = dict(ckpt=a.ckpt, store=a.store, positions=n, engines={})
    for e, r in res.items():
        info["engines"][e] = dict(max_abs_dp=max(r["maxdp"]), mean_max_abs_dp=float(np.mean(r["maxdp"])),
                                  mean_kl=float(np.mean(r["kl"])), top_move_agree=float(np.mean(r["top"])))
    Path(a.out).write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
