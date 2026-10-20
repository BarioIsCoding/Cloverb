"""Attention microbenchmark at Cloverb's training shape: PyTorch SDPA vs. the fused attn64 kernel.

Times forward and forward+backward on q, k, v [B, H, 64, D] with a static [H, 64, 64] bias, and checks
the kernel against the PyTorch reference in fp32 (output and all gradients).
FLOPs count multiplies and adds: forward 4*64*64*D per head, backward 8*64*64*D (score recompute not counted).

usage: python tools/bench_attention.py [--batch 512] [--threads 4] --out results/attn_bench.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cloverb.attn import attn64, reference  # noqa: E402


def timeit(fn, reps):
    for _ in range(3):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    B, H, L, D = a.batch, a.heads, 64, a.dim
    scale = D ** -0.5
    fwd_flop, bwd_flop = 4 * B * H * L * L * D, 8 * B * H * L * L * D
    stat = torch.randn(H, L, L)
    coef, tmpl = torch.zeros(B, H, 0), torch.zeros(0, L * L)

    # correctness in fp32 against the PyTorch reference
    q, k, v = (torch.randn(B, H, L, D, requires_grad=True) for _ in range(3))
    st = stat.clone().requires_grad_(True)
    o = attn64(q, k, v, st, coef, tmpl, scale)
    g = torch.randn_like(o)
    gk = torch.autograd.grad(o, (q, k, v, st), g)
    o_ref = reference(q, k, v, st, coef, tmpl, scale)
    gr = torch.autograd.grad(o_ref, (q, k, v, st), g)
    err = dict(out=(o - o_ref).abs().max().item(),
               **{f"grad_{n}": ((x - y).abs().max() / y.abs().max()).item() for n, x, y in zip("qkvb", gk, gr)})

    res = dict(shape=[B, H, L, D], threads=a.threads, fp32_max_abs_err_out=err["out"],
               fp32_max_rel_err_grads={k_: v_ for k_, v_ in err.items() if k_ != "out"}, timings={})
    for dt in (torch.float32, torch.bfloat16):
        q, k, v = (torch.randn(B, H, L, D, dtype=dt, requires_grad=True) for _ in range(3))
        st = stat.clone().requires_grad_(True)
        g = torch.randn(B, H, L, D, dtype=dt)
        mask = st.to(dt)
        impl = {
            "sdpa": lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=mask.unsqueeze(0), scale=scale),
            "attn64": lambda: attn64(q, k, v, st, coef, tmpl, scale),
        }
        for name, f in impl.items():
            with torch.no_grad():
                tf = timeit(f, a.reps)

            def fb():
                q.grad = k.grad = v.grad = st.grad = None
                f().backward(g)
            tfb = timeit(fb, a.reps)
            res["timings"][f"{name}_{str(dt).split('.')[-1]}"] = dict(
                fwd_ms=1e3 * tf, fwd_bwd_ms=1e3 * tfb, fwd_gflops=fwd_flop / tf / 1e9,
                fwd_bwd_gflops=(fwd_flop + bwd_flop) / tfb / 1e9)
    Path(a.out).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
