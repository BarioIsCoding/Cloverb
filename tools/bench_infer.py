"""Inference speed: Cloverb vs Maia-3 (official code and our fast re-implementation).

Measures throughput at batch 256 (4 threads) and single-position latency (batch 1, 1 thread),
end to end from tensors (input building excluded; it is ~4 us/position in C for Cloverb).
usage: python tools/bench_infer.py --cloverb runs/X/ema.pt [--out results/infer_speed.json]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))


def timeit(f, n):
    f(); f()
    t = time.perf_counter()
    for _ in range(n):
        f()
    return (time.perf_counter() - t) / n


def cloverb_inputs(B, H=64):
    board = torch.randint(0, 13, (B, 64), dtype=torch.int8)
    meta = torch.zeros(B, 8, dtype=torch.int16); meta[:, 4] = -1; meta[:, 6] = 20
    hist = torch.randint(0, 64, (B, H, 4), dtype=torch.int8); hist[..., 2] %= 6; hist[..., 3] %= 2
    f = lambda v: torch.full((B,), float(v))
    return board, meta, hist, f(1500), f(180), f(2), f(100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloverb", required=True)
    ap.add_argument("--out", default=str(ROOT / "results" / "infer_speed.json"))
    a = ap.parse_args()
    from cloverb.infer import load
    from cloverb.teacher import FastMaia3
    import bench_maia3 as bm
    res = {}
    models = {"cloverb": lambda: load(a.cloverb)}
    for v in ["maia3-5m", "maia3-23m", "maia3-79m"]:
        models[v + " (official)"] = (lambda v=v: bm.load_variant(v)[0])
        models[v + " (fast)"] = (lambda v=v: FastMaia3(v))
    for name, mk in models.items():
        m = mk().eval()
        for threads, B, n in [(4, 256, 5), (1, 1, 50)]:
            torch.set_num_threads(threads)
            if name == "cloverb":
                x = cloverb_inputs(B)
                fn = lambda: m(*x)
            else:
                xx = torch.randint(0, 2, (B, 64, 96)).float(); e = torch.full((B,), 1500)
                fn = lambda: m(xx, e, e)
            for prec in ["fp32", "bf16"]:
                def run():
                    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=prec == "bf16"):
                        fn()
                dt = timeit(run, n)
                key = f"{name} | {prec} | batch {B} | {threads} thr"
                res[key] = dict(ms=dt * 1000, pos_per_s=B / dt)
                print(f"{key:55s} {dt * 1000:8.2f} ms  {B / dt:9.0f} pos/s", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
