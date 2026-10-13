"""Export a Cloverb checkpoint for the standalone C engine (csrc/cloverb_engine.c).

Format (little endian): b"CLVB", int32 version=1, int32 n_cfg, n_cfg x int32 config values
(d, layers, heads, mlp_hidden, hist, d_cond, n_templates, static_bias, adaln, postln, cap_type,
value_hidden, time_bins, act_relu2, n_cond), int32 n_tensors, then per tensor:
int32 name_len, name bytes, int32 ndim, ndim x int32 dims, float32 data (row-major).
usage: python tools/export_engine.py CKPT OUT.bin
"""
import struct
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloverb.model import N_COND, TIME_BINS, Cloverb, CloverbConfig  # noqa: E402


def main():
    st = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    cfg = CloverbConfig(**st["cfg"])
    assert not cfg.opp_elo, "the engine conditions on the player's Elo only"
    m = Cloverb(cfg)
    m.load_state_dict(st["ema"] if "ema" in st else st["model"])
    vals = [cfg.d, cfg.layers, cfg.heads, int(cfg.mlp_mult * cfg.d), cfg.hist, cfg.d_cond, cfg.n_templates,
            int(cfg.static_bias), int(cfg.adaln), int(cfg.postln), int(cfg.cap_type), cfg.value_hidden, TIME_BINS,
            int(cfg.act == "relu2"), N_COND]
    sd = {k: v.float().contiguous() for k, v in m.state_dict().items()}
    with open(sys.argv[2], "wb") as f:
        f.write(b"CLVB"); f.write(struct.pack("<ii", 1, len(vals))); f.write(struct.pack(f"<{len(vals)}i", *vals))
        f.write(struct.pack("<i", len(sd)))
        for k, v in sd.items():
            nb = k.encode()
            f.write(struct.pack("<i", len(nb))); f.write(nb)
            f.write(struct.pack("<i", v.dim())); f.write(struct.pack(f"<{v.dim()}i", *v.shape))
            f.write(v.numpy().tobytes())
    print(f"exported {len(sd)} tensors, {sum(v.numel() for v in sd.values()) / 1e6:.2f}M floats -> {sys.argv[2]}")


if __name__ == "__main__":
    main()
