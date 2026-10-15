"""Score released Maia-3 checkpoints on a Cloverb game store with the official model code.

Inputs are built exactly as the authors' ``maia3.uci --use-uci-history`` does: the last
``history`` board tokenizations, each mirrored to its *own* side to move, earliest one
repeated as padding; self and opponent Elo; clock channels only for the ponder model.

Per scored ply we save the legal-renormalized cross-entropy and the rank of the played
move among legal moves (top-1 = rank 0, top-3 = rank < 3).

usage: python tools/bench_maia3.py --variant maia3-5m --store S.npz --prep P.npz --out R.npz
       [--subset SUBSET.npy] [--protocol maia|all] [--fp32] [--history-mode own|none]
"""
import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/home/user/ext/maia3")
from cloverb.gamestore import GameStore, maia_protocol_mask  # noqa: E402
from maia3.models import MAIA3Model  # noqa: E402
from maia3.model_registry import resolve_model_spec  # noqa: E402

MODELS = Path("/home/user/models/maia3")
VARIANTS = {
    "maia3-3m": ("maia3-3m-ablation", "maia3-3m.pt", {}),
    "maia3-5m": ("maia3-5m", "maia3-5m.pt", {}),
    "maia3-23m": ("maia3-23m", "maia3-23m.pt", {}),
    "maia3-79m": ("maia3-79m", "maia3-79m.pt", {}),
    "hist1": ("maia3-5m", "hist1.pt", {"history": 1}),
    "hist32": ("maia3-5m", "hist32.pt", {"history": 32}),
    "abspos": ("maia3-5m", "abs_pe.pt", {"use_gab": False, "use_absolute_pe": True}),
    "relpos": ("maia3-5m", "rel_pe.pt", {"use_gab": False, "use_relative_bias": True}),
    "maia3-23m-ponder": ("maia3-23m", "maia3-23m-ponder.pt", {"include_time_info": True, "ponder_bins": 30}),
}


def load_variant(name):
    spec_name, fname, over = VARIANTS[name]
    cfg = Namespace(**{**resolve_model_spec(spec_name).config, **over})
    model = MAIA3Model(cfg)
    if getattr(cfg, "ponder_bins", 1) != 1:  # the released code cannot build this head
        model.fc_ponder = torch.nn.Linear(cfg.head_hid_dim, cfg.ponder_bins)
    sd = torch.load(MODELS / fname, map_location="cpu", weights_only=True)
    sd = sd.get("model_state_dict", sd)
    sd = {k.replace("smolgen", "gab"): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing, missing
    assert not unexpected, unexpected
    model.eval()
    uniq = {id(p): p.numel() for p in model.parameters()}
    return model, cfg, sum(uniq.values())


def policy_logits(model, x, self_elo, oppo_elo, bf16: bool):
    """MAIA3Model.forward's policy path, trunk optionally in bf16 but the from-to head in fp32:
    bf16 logits tie often enough to inflate top-1 by ~1 point on CPU."""
    cfg = model.cfg
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        width = 12 * cfg.history + (3 if cfg.include_time_info else 0)
        x = x[:, :, :width]
        se = model.interpolate_elo(self_elo).unsqueeze(1).expand(-1, 64, -1)
        oe = model.interpolate_elo(oppo_elo).unsqueeze(1).expand(-1, 64, -1)
        h = model.token_projection(torch.cat([x, se, oe], -1))
        if hasattr(model, "abs_pe"):
            h = model.abs_pe(h)
        h = model.transformer(h)
    h = h.float()
    sq_from = h @ model.proj_sq_from.weight.float().T
    sq_to = h @ model.proj_sq_to.weight.float().T
    d = cfg.head_hid_dim
    base = torch.einsum("bid,bjd->bij", sq_from, sq_to) / d ** 0.5
    promo = (sq_to[:, 56:64] @ model.promo_bias_proj.weight.float().T) * d ** 0.5  # B,8(to file),4
    pb = base[:, 48:56, 56:64]  # B, from file, to file
    promo_logits = (pb[:, :, :, None] + promo[:, None, :, :]).reshape(x.shape[0], 256)
    return torch.cat([base.reshape(x.shape[0], 4096), promo_logits], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--prep", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset", default=None, help="npy of global ply indices to score")
    ap.add_argument("--protocol", default="maia", choices=["maia", "all"])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--history-mode", default="own", choices=["own", "none"],
                    help="none = repeat the current position (the UCI default without --use-uci-history)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--time-scale", type=float, default=100.0)
    ap.add_argument("--fast", action="store_true", help="use cloverb.teacher.FastMaia3 (same weights, compiled)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    gs = GameStore.load(a.store)
    pz = np.load(a.prep)
    planes, target, legal, loff = pz["planes"], pz["target"].astype(np.int64), pz["legal"], pz["legal_off"]
    if a.subset:
        idx = np.load(a.subset)
    else:
        mask = maia_protocol_mask(gs) if a.protocol == "maia" else np.ones(len(gs.moves), bool)
        idx = np.flatnonzero(mask)
    game_of = np.searchsorted(gs.offsets, idx, side="right") - 1
    start = gs.offsets[game_of]
    white = ((idx - start) % 2 == 0)
    self_elo = np.where(white, gs.welo[game_of], gs.belo[game_of]).astype(np.int64)
    oppo_elo = np.where(white, gs.belo[game_of], gs.welo[game_of]).astype(np.int64)

    model, cfg, nparams = load_variant(a.variant)
    fast = None
    if a.fast:
        from cloverb.teacher import CONFIGS, FastMaia3
        assert a.variant in CONFIGS, f"no fast path for {a.variant}"
        fast = torch.compile(FastMaia3(a.variant))
    H = cfg.history
    ce = np.zeros(len(idx), np.float32)
    rank = np.zeros(len(idx), np.int16)
    t0 = time.perf_counter(); fwd = 0.0
    for s in range(0, len(idx), a.batch):
        sl = slice(s, s + a.batch)
        k = idx[sl]; B = len(k)
        if a.history_mode == "own":
            h = k[:, None] - np.arange(H - 1, -1, -1)[None, :]
            h = np.maximum(h, start[sl][:, None])
        else:
            h = np.repeat(k[:, None], H, 1)
        bits = np.unpackbits(planes[h].view(np.uint8), axis=-1, bitorder="little").reshape(B, H, 12, 64)
        x = torch.from_numpy(bits.transpose(0, 3, 1, 2).reshape(B, 64, H * 12).astype(np.float32))
        if cfg.include_time_info:
            g = game_of[sl]
            tf = np.stack([gs.base[g], gs.inc[g], gs.clock[k], np.zeros(B)], 1).astype(np.float32) / a.time_scale
            x = torch.cat([x, torch.from_numpy(tf)[:, None, :].expand(B, 64, 4)], 2)
        f0 = time.perf_counter()
        with torch.inference_mode():
            if fast is not None:
                tm = None
                if cfg.include_time_info:
                    g = game_of[sl]
                    tm = torch.from_numpy(np.stack([gs.base[g], gs.inc[g], gs.clock[k]], 1).astype(np.float32))
                with torch.autocast("cpu", dtype=torch.bfloat16, enabled=not a.fp32):
                    logits = fast(x[:, :, :H * 12], torch.from_numpy(self_elo[sl]), torch.from_numpy(oppo_elo[sl]), tm).float()
            else:
                logits = policy_logits(model, x, torch.from_numpy(self_elo[sl]), torch.from_numpy(oppo_elo[sl]),
                                       bf16=not a.fp32)
        fwd += time.perf_counter() - f0
        lm = torch.zeros(B, logits.shape[1], dtype=torch.bool)
        rows = np.repeat(np.arange(B), loff[k + 1] - loff[k])
        cols = np.concatenate([legal[loff[i]:loff[i + 1]] for i in k]).astype(np.int64)
        lm[torch.from_numpy(rows), torch.from_numpy(cols)] = True
        logits = logits.masked_fill(~lm, float("-inf"))
        t = torch.from_numpy(target[k])
        tl = logits.gather(1, t[:, None])
        ce[sl] = (torch.logsumexp(logits, 1) - tl[:, 0]).numpy()
        rank[sl] = (logits > tl).sum(1).numpy()
        if s // a.batch % 200 == 0:
            done = s + B
            print(f"{a.variant}: {done}/{len(idx)} top1={np.mean(rank[:done] == 0):.4f} "
                  f"{done / (time.perf_counter() - t0):.0f} pos/s", flush=True)
    dt = time.perf_counter() - t0
    top1, top3 = float(np.mean(rank == 0)), float(np.mean(rank < 3))
    se = lambda p: 1.96 * np.sqrt(p * (1 - p) / len(idx))
    info = dict(variant=a.variant, params=int(nparams), positions=int(len(idx)), top1=top1, top1_ci=se(top1),
                top3=top3, top3_ci=se(top3), ce=float(ce.mean()), history=H, history_mode=a.history_mode,
                dtype="fp32" if a.fp32 else "bf16", seconds=dt, forward_seconds=fwd, pos_per_s=len(idx) / dt,
                store=a.store, subset=a.subset, protocol="subset" if a.subset else a.protocol)
    np.savez(a.out, idx=idx, ce=ce, rank=rank)
    Path(a.out).with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
