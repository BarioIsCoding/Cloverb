"""torch custom op around csrc/attn64.c: fused 64-token attention with static + template-mixed bias.

attn64(q, k, v, stat, coef, tmpl, scale) -> o
  q, k, v  [B, H, 64, D]  (bf16 or fp32)       stat [H, 64, 64] fp32
  coef     [B, H, T] fp32 (T may be 0)         tmpl [T, 64*64] fp32
  o = softmax(scale * q k^T + stat[h] + sum_t coef[b,h,t] tmpl[t]) v
Registered with torch.library so torch.compile keeps it as one opaque kernel.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "csrc" / "attn64.c"
LIB = HERE / "libattn64.so"


def _load():
    if SRC.exists() and (not LIB.exists() or LIB.stat().st_mtime < SRC.stat().st_mtime):
        tmp = LIB.with_suffix(f".{os.getpid()}.tmp")
        subprocess.check_call(["gcc", "-O3", "-march=native", "-fopenmp", "-shared", "-fPIC", "-o", str(tmp), str(SRC), "-lm"])
        os.replace(tmp, LIB)
    lib = ctypes.CDLL(str(LIB))
    vp, i, f = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
    lib.attn64_fwd.argtypes = [vp, vp, vp, vp, vp, vp, i, i, i, i, f, i, vp]
    lib.attn64_bwd.argtypes = [vp, vp, vp, vp, vp, vp, vp, vp, i, i, i, i, f, i, vp, vp, vp, vp, vp, vp, vp]
    lib.attn64_fwd.restype = lib.attn64_bwd.restype = lib.attn64_threads.restype = i
    return lib


_lib = _load()


def _ptr(t):
    return ctypes.c_void_p(t.data_ptr()) if t.numel() else None


def _prep(q, k, v):
    dt = q.dtype if q.dtype in (torch.bfloat16, torch.float32) else torch.float32
    return [x.to(dt).contiguous() for x in (q, k, v)], dt


@torch.library.custom_op("cloverb::attn64", mutates_args=())
def attn64(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, stat: torch.Tensor, coef: torch.Tensor,
           tmpl: torch.Tensor, scale: float) -> torch.Tensor:
    (q, k, v), dt = _prep(q, k, v)
    B, H, L, D = q.shape
    T = tmpl.shape[0]
    stat, coef, tmpl = stat.float().contiguous(), coef.float().contiguous(), tmpl.float().contiguous()
    o = torch.empty_like(q)
    rc = _lib.attn64_fwd(_ptr(q), _ptr(k), _ptr(v), _ptr(stat), _ptr(coef), _ptr(tmpl), B * H, H, D, T, scale,
                         int(dt == torch.bfloat16), _ptr(o))
    assert rc == 0
    return o


def _dt(q):
    return q.dtype if q.dtype in (torch.bfloat16, torch.float32) else torch.float32


@attn64.register_fake
def _(q, k, v, stat, coef, tmpl, scale):
    return q.new_empty(q.shape, dtype=_dt(q))  # contiguous, like the real kernel


@torch.library.custom_op("cloverb::attn64_bwd", mutates_args=())
def attn64_bwd(do: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor,
               stat: torch.Tensor, coef: torch.Tensor, tmpl: torch.Tensor, scale: float
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    (q, k, v), dt = _prep(q, k, v)
    o, do = o.to(dt).contiguous(), do.to(dt).contiguous()
    B, H, L, D = q.shape
    T = tmpl.shape[0]
    stat, coef, tmpl = stat.float().contiguous(), coef.float().contiguous(), tmpl.float().contiguous()
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    dstat = torch.empty(H, L, L, dtype=torch.float32)
    dcoef = torch.empty(B, H, T, dtype=torch.float32)
    dtmpl = torch.empty(T, L * L, dtype=torch.float32)
    partial = torch.empty(_lib.attn64_threads() * (H + T) * L * L, dtype=torch.float32)
    rc = _lib.attn64_bwd(_ptr(q), _ptr(k), _ptr(v), _ptr(o), _ptr(do), _ptr(stat), _ptr(coef), _ptr(tmpl),
                         B * H, H, D, T, scale, int(dt == torch.bfloat16), _ptr(dq), _ptr(dk), _ptr(dv),
                         _ptr(dstat), _ptr(dcoef), _ptr(dtmpl), _ptr(partial))
    assert rc == 0
    return dq, dk, dv, dstat, dcoef, dtmpl


@attn64_bwd.register_fake
def _(do, q, k, v, o, stat, coef, tmpl, scale):
    B, H, L, D = q.shape
    T = tmpl.shape[0]
    dt = _dt(q)
    return (q.new_empty(q.shape, dtype=dt), q.new_empty(k.shape, dtype=dt), q.new_empty(v.shape, dtype=dt),
            q.new_empty(H, L, L, dtype=torch.float32),
            q.new_empty(B, H, T, dtype=torch.float32), q.new_empty(T, L * L, dtype=torch.float32))


def _setup(ctx, inputs, output):
    q, k, v, stat, coef, tmpl, scale = inputs
    ctx.save_for_backward(q, k, v, output, stat, coef, tmpl)
    ctx.scale = scale
    ctx.dtypes = (q.dtype, k.dtype, v.dtype, stat.dtype, coef.dtype, tmpl.dtype)


def _backward(ctx, do):
    q, k, v, o, stat, coef, tmpl = ctx.saved_tensors
    dq, dk, dv, dstat, dcoef, dtmpl = attn64_bwd(do, q, k, v, o, stat, coef, tmpl, ctx.scale)
    dts = ctx.dtypes
    return dq.to(dts[0]), dk.to(dts[1]), dv.to(dts[2]), dstat.to(dts[3]), dcoef.to(dts[4]), dtmpl.to(dts[5]), None


attn64.register_autograd(_backward, setup_context=_setup)


def reference(q, k, v, stat, coef, tmpl, scale):
    B, H = q.shape[:2]
    bias = stat.unsqueeze(0) + (coef @ tmpl).view(B, H, 64, 64) if tmpl.shape[0] else stat.unsqueeze(0).expand(B, -1, -1, -1)
    s = (q.float() @ k.float().transpose(-1, -2)) * scale + bias
    return (torch.softmax(s, -1) @ v.float()).to(q.dtype)


# ------------------------------------------------------------------------------------------------
# qkv-fused variant: takes the raw projection [B, 64, 3d], applies per-head RMSNorm (weights wq, wk
# shared across heads) to q and k inside the kernel and returns o as [B, 64, d].
_lib.attn64qkv_fwd.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 4 + [ctypes.c_float] * 2 + [ctypes.c_int, ctypes.c_void_p]
_lib.attn64qkv_bwd.argtypes = ([ctypes.c_void_p] * 8 + [ctypes.c_int] * 4 + [ctypes.c_float] * 2 + [ctypes.c_int]
                               + [ctypes.c_void_p] * 7)
_lib.attn64qkv_fwd.restype = _lib.attn64qkv_bwd.restype = ctypes.c_int


@torch.library.custom_op("cloverb::attn64qkv", mutates_args=())
def attn64qkv(qkv: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor, stat: torch.Tensor, coef: torch.Tensor,
              tmpl: torch.Tensor, heads: int, scale: float, eps: float) -> torch.Tensor:
    dt = _dt(qkv)
    qkv = qkv.to(dt).contiguous()
    B, L, d3 = qkv.shape
    d = d3 // 3
    D = d // heads
    T = tmpl.shape[0]
    wq, wk = wq.float().contiguous(), wk.float().contiguous()
    stat, coef, tmpl = stat.float().contiguous(), coef.float().contiguous(), tmpl.float().contiguous()
    o = qkv.new_empty(B, L, d)
    rc = _lib.attn64qkv_fwd(_ptr(qkv), _ptr(wq), _ptr(wk), _ptr(stat), _ptr(coef), _ptr(tmpl), B, heads, D, T, scale,
                            eps, int(dt == torch.bfloat16), _ptr(o))
    assert rc == 0
    return o


@attn64qkv.register_fake
def _(qkv, wq, wk, stat, coef, tmpl, heads, scale, eps):
    B, L, d3 = qkv.shape
    return qkv.new_empty(B, L, d3 // 3, dtype=_dt(qkv))


@torch.library.custom_op("cloverb::attn64qkv_bwd", mutates_args=())
def attn64qkv_bwd(do: torch.Tensor, qkv: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor, o: torch.Tensor,
                  stat: torch.Tensor, coef: torch.Tensor, tmpl: torch.Tensor, heads: int, scale: float, eps: float
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dt = _dt(qkv)
    qkv, o, do = qkv.to(dt).contiguous(), o.to(dt).contiguous(), do.to(dt).contiguous()
    B, L, d3 = qkv.shape
    d = d3 // 3
    D = d // heads
    T = tmpl.shape[0]
    wq, wk = wq.float().contiguous(), wk.float().contiguous()
    stat, coef, tmpl = stat.float().contiguous(), coef.float().contiguous(), tmpl.float().contiguous()
    dqkv = torch.empty_like(qkv)
    dwq, dwk = torch.empty(D), torch.empty(D)
    dstat = torch.empty(heads, L, L)
    dcoef = torch.empty(B, heads, T)
    dtmpl = torch.empty(T, L * L)
    partial = torch.empty(_lib.attn64_threads() * ((heads + T) * L * L + 2 * D))
    rc = _lib.attn64qkv_bwd(_ptr(qkv), _ptr(wq), _ptr(wk), _ptr(o), _ptr(do), _ptr(stat), _ptr(coef), _ptr(tmpl),
                            B, heads, D, T, scale, eps, int(dt == torch.bfloat16), _ptr(dqkv), _ptr(dwq), _ptr(dwk),
                            _ptr(dstat), _ptr(dcoef), _ptr(dtmpl), _ptr(partial))
    assert rc == 0
    return dqkv, dwq, dwk, dstat, dcoef, dtmpl


@attn64qkv_bwd.register_fake
def _(do, qkv, wq, wk, o, stat, coef, tmpl, heads, scale, eps):
    B, L, d3 = qkv.shape
    D = d3 // 3 // heads
    T = tmpl.shape[0]
    f = lambda *s: qkv.new_empty(*s, dtype=torch.float32)
    return qkv.new_empty(qkv.shape, dtype=_dt(qkv)), f(D), f(D), f(heads, L, L), f(B, heads, T), f(T, L * L)


def _setup_qkv(ctx, inputs, output):
    qkv, wq, wk, stat, coef, tmpl, heads, scale, eps = inputs
    ctx.save_for_backward(qkv, wq, wk, output, stat, coef, tmpl)
    ctx.args = (heads, scale, eps)
    ctx.dtypes = (qkv.dtype, wq.dtype, wk.dtype, stat.dtype, coef.dtype, tmpl.dtype)


def _backward_qkv(ctx, do):
    qkv, wq, wk, o, stat, coef, tmpl = ctx.saved_tensors
    grads = attn64qkv_bwd(do, qkv, wq, wk, o, stat, coef, tmpl, *ctx.args)
    return tuple(g.to(t) for g, t in zip(grads, ctx.dtypes)) + (None, None, None)


attn64qkv.register_autograd(_backward_qkv, setup_context=_setup_qkv)
