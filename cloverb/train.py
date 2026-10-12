"""Train Cloverb on Lichess game stores (CPU, bf16 autocast on AMX).

usage: python -m cloverb.train --stores S1.npz S2.npz ... --out runs/NAME [--steps N] [...]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import MODEL_KEYS, PoolSampler, Prefetcher, Sampler, SamplerConfig, concat_stores, make_inputs
from .gamestore import GameStore
from .model import Cloverb, CloverbConfig, count_params, time_bucket


def lr_at(step, a):
    """Warmup, constant, then linear decay to `a.lr_min_frac` over the last `a.decay_frac` of training."""
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    d0 = int(a.steps * (1 - a.decay_frac))
    if step < d0:
        return a.lr
    t = (step - d0) / max(1, a.steps - d0)
    return a.lr * (1 - t * (1 - a.lr_min_frac))


def maia_extra(gs, g, ply):
    from .baseline import maia_inputs
    x, se, oe = maia_inputs(gs, g, ply)
    return dict(maia_x=x, se=se, oe=oe)


def run_model(model, x, arch):
    if arch == "chessformer":
        lo, v = model(x["maia_x"], x["se"], x["oe"])
        return lo, v, None
    m = getattr(model, "_orig_mod", model)
    if getattr(getattr(m, "cfg", None), "opp_elo", False):
        return model(*(x[k] for k in MODEL_KEYS), x["oelo"])
    return model(*(x[k] for k in MODEL_KEYS))


@torch.no_grad()
def evaluate(model, gs: GameStore, idx: np.ndarray, B: int = 1024, H: int = 64, arch: str = "cloverb"):
    model.eval()
    g = np.searchsorted(gs.offsets, idx, side="right") - 1
    ply = idx - gs.offsets[g]
    ce, rank = [], []
    for s in range(0, len(idx), B):
        x = make_inputs(gs, g[s:s + B], ply[s:s + B], H)
        if arch == "chessformer":
            x.update(maia_extra(gs, g[s:s + B], ply[s:s + B]))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            lo = run_model(model, x, arch)[0]
        lo = lo.float().masked_fill(~x["legal"], float("-inf"))
        t = x["target"]
        tl = lo.gather(1, t[:, None])
        ce.append((torch.logsumexp(lo, 1) - tl[:, 0]).numpy())
        rank.append((lo > tl).sum(1).numpy())
    model.train()
    ce, rank = np.concatenate(ce), np.concatenate(rank)
    return dict(top1=float(np.mean(rank == 0)), top3=float(np.mean(rank < 3)), ce=float(ce.mean())), ce, rank


def param_groups(model, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (decay if p.ndim >= 2 and "static" not in n and "templates" not in n and "square" not in n else no_decay).append(p)
    return [dict(params=decay, weight_decay=wd), dict(params=no_decay, weight_decay=0.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stores", nargs="+", required=True)
    ap.add_argument("--arch", default="cloverb", choices=["cloverb", "chessformer"])
    ap.add_argument("--cf-size", default="5m")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-store", default="/home/user/data/bench/allie2022.npz")
    ap.add_argument("--eval-idx", default="/home/user/data/bench/allie_sub10k.npy")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--mlp", type=float, default=3.0)
    ap.add_argument("--templates", type=int, default=32)
    ap.add_argument("--no-static", action="store_true")
    ap.add_argument("--no-adaln", action="store_true")
    ap.add_argument("--act", default="relu2", choices=["relu2", "gelu"])
    ap.add_argument("--geo-init", action="store_true")
    ap.add_argument("--small-init", action="store_true", help="small normal instead of zero init of residual outputs")
    ap.add_argument("--cap-type", action="store_true", help="history encodes the captured piece type")
    ap.add_argument("--postln", action="store_true", help="post-norm blocks")
    ap.add_argument("--opp-elo", action="store_true", help="diagnostic: condition on the opponent's Elo too")
    ap.add_argument("--hist", type=int, default=64)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-min-frac", type=float, default=0.02)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--decay-frac", type=float, default=0.25)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--value-w", type=float, default=0.1)
    ap.add_argument("--time-w", type=float, default=0.05)
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--elo-alpha", type=float, default=0.3)
    ap.add_argument("--max-hours", type=float, default=1e9)
    ap.add_argument("--teacher", default=None, help="maia3-5m | maia3-23m | maia3-79m (online distillation)")
    ap.add_argument("--distill-w", type=float, default=0.5, help="weight of teacher soft-CE vs human-move CE")
    ap.add_argument("--teacher-cache", nargs="*", default=None, help="teacher-labelled pools (label_teacher.py)")
    ap.add_argument("--distill-w-pressure", type=float, default=None,
                    help="distillation weight when the mover has < 30 s (Maia-3 never trained there)")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = CloverbConfig(d=a.d, layers=a.layers, heads=a.heads, mlp_mult=a.mlp, hist=a.hist, n_templates=a.templates,
                        static_bias=not a.no_static, adaln=not a.no_adaln, act=a.act, geo_init=a.geo_init,
                        opp_elo=a.opp_elo, zero_init=not a.small_init, cap_type=a.cap_type,
                        postln=a.postln)
    if a.arch == "chessformer":
        from .baseline import Chessformer
        model = Chessformer(a.cf_size)
    else:
        model = Cloverb(cfg)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(param_groups(model, a.wd), lr=a.lr, betas=(0.9, 0.98), eps=1e-8)
    step = 0
    ck = out / "last.pt"
    if ck.exists():
        st = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); ema.load_state_dict(st["ema"]); opt.load_state_dict(st["opt"])
        step = st["step"]
        print(f"resumed at step {step}", flush=True)
    (out / "config.json").write_text(json.dumps(dict(model=cfg.to_dict(), args=vars(a)), indent=2))
    print(f"params {count_params(model) / 1e6:.2f}M", flush=True)

    gs = concat_stores(a.stores)
    print(f"train store: {gs.n_games} games, {len(gs.moves)} plies", flush=True)
    if a.teacher_cache:
        sampler = PoolSampler(a.teacher_cache, a.stores, seed=a.seed + step, n_games=gs.n_games)
        print(f"teacher pool: {len(sampler.g)} positions", flush=True)
    else:
        sampler = Sampler(gs, SamplerConfig(elo_alpha=a.elo_alpha, seed=a.seed + step))
    need_maia = a.arch == "chessformer" or a.teacher is not None
    pf = Prefetcher(gs, sampler, a.batch, a.hist, extra=maia_extra if need_maia else None)
    teacher = None
    if a.teacher:
        from .teacher import FastMaia3, teacher_logprobs
        teacher = torch.compile(FastMaia3(a.teacher)) if a.compile else FastMaia3(a.teacher)
    egs = GameStore.load(a.eval_store)
    eidx = np.load(a.eval_idx)

    fwd = torch.compile(model) if a.compile else model
    log = open(out / "log.jsonl", "a")
    t0 = time.time(); tl = time.time()
    acc_loss = acc_top1 = acc_n = 0.0
    while step < a.steps:
        for gparam in opt.param_groups:
            gparam["lr"] = lr_at(step, a)
        x = next(pf)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            lo, v, t = run_model(fwd, x, a.arch)
        lo = lo.float().masked_fill(~x["legal"], float("-inf"))
        lp = F.cross_entropy(lo, x["target"])
        if teacher is not None or "soft" in x:
            if "soft" in x:
                pt = x["soft"]
            else:
                tm = torch.stack([x["base"], x["inc"], x["clock"]], 1)
                pt = teacher_logprobs(teacher, x["maia_x"], x["se"], x["oe"], x["legal"], time=tm).exp()
            ls_i = -(pt * torch.log_softmax(lo, 1).masked_fill(~x["legal"], 0)).sum(1)
            lp_i = F.cross_entropy(lo, x["target"], reduction="none")
            w = torch.full_like(ls_i, a.distill_w)
            if a.distill_w_pressure is not None:
                w = torch.where(x["clock"] < 30, torch.full_like(w, a.distill_w_pressure), w)
            lpol = ((1 - w) * lp_i + w * ls_i).mean()
        else:
            lpol = lp
        lv = F.cross_entropy(v.float(), x["wdl"])
        tm = x["spent"] >= 0
        lt = F.cross_entropy(t.float()[tm], time_bucket(x["spent"][tm])) if t is not None and tm.any() else lo.new_zeros(())
        loss = lpol + a.value_w * lv + a.time_w * lt
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step()
        step += 1
        with torch.no_grad():
            dec = min(a.ema, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - dec)
            acc_loss += lp.item(); acc_top1 += (lo.argmax(1) == x["target"]).float().mean().item(); acc_n += 1
        if step % a.log_every == 0:
            dt = time.time() - tl; tl = time.time()
            rec = dict(step=step, loss=acc_loss / acc_n, top1=acc_top1 / acc_n, value=lv.item(), time=lt.item(),
                       gnorm=float(gn), lr=lr_at(step, a), pos_s=a.log_every * a.batch / dt,
                       hours=(time.time() - t0) / 3600)
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
            acc_loss = acc_top1 = acc_n = 0.0
        if step % a.eval_every == 0 or step == a.steps:
            r, _, _ = evaluate(ema, egs, eidx, arch=a.arch)
            r0, _, _ = evaluate(model, egs, eidx, arch=a.arch)
            rec = dict(step=step, eval_ema=r, eval_raw=r0, positions_seen=step * a.batch)
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        if step % a.ckpt_every == 0 or step == a.steps or (time.time() - t0) / 3600 > a.max_hours:
            tmp = out / "last.pt.tmp"
            torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), step=step,
                            cfg=cfg.to_dict(), arch=a.arch), tmp)
            os.replace(tmp, ck)
            torch.save(dict(ema=ema.state_dict(), cfg=cfg.to_dict(), step=step, arch=a.arch), out / "ema.pt")
            if (time.time() - t0) / 3600 > a.max_hours:
                break
    pf.close()


if __name__ == "__main__":
    main()
