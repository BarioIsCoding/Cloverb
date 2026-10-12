"""Stream the beginning of a Lichess .pgn.zst dump into a game store (no full download).

usage: python tools/pgn_stream_store.py URL OUT.npz --games 200000 [--skip 0]
Same filters as tools/build_store.py.
"""
import argparse
import io
import sys
import urllib.request
from pathlib import Path

import numpy as np
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloverb import core  # noqa: E402
from cloverb.gamestore import GameStore  # noqa: E402

RESULT = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}


def games(stream):
    hdr, mt = {}, []
    for line in stream:
        line = line.rstrip("\n")
        if line.startswith("["):
            if mt:
                yield hdr, " ".join(mt); hdr, mt = {}, []
            k, _, v = line[1:-1].partition(" ")
            hdr[k] = v.strip('"')
        elif line:
            mt.append(line)
    if mt:
        yield hdr, " ".join(mt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url"); ap.add_argument("out")
    ap.add_argument("--games", type=int, default=200000)
    a = ap.parse_args()
    req = urllib.request.urlopen(a.url)
    text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(req), encoding="utf-8")
    txt, meta, seen = [], [], 0
    for h, mt in games(text):
        seen += 1
        tc, term = h.get("TimeControl", "-"), h.get("Termination", "")
        if "+" not in tc or term not in ("Normal", "Time forfeit") or h.get("Result") not in RESULT or "%clk" not in mt:
            continue
        if not h.get("WhiteElo", "?").isdigit() or not h.get("BlackElo", "?").isdigit():
            continue
        base, inc = map(int, tc.split("+"))
        if base + 40 * inc < 30:
            continue
        txt.append(mt); meta.append((int(h["WhiteElo"]), int(h["BlackElo"]), base, inc, RESULT[h["Result"]]))
        if len(txt) >= a.games:
            break
    mv, ca, nply = core.parse_games(txt)
    out_m, out_c, out_s, lens, gm, off = [], [], [], [], [], 0
    for (we, be, base, inc, res), n in zip(meta, nply):
        if n < 0:
            continue
        m, c = mv[off:off + n], ca[off:off + n]
        off += n
        if n < 2 or (c < 0).any():
            continue
        before = np.empty(n, np.int64); before[:2] = base; before[2:] = c[:-2]
        out_m.append(m); out_c.append(np.clip(before, 0, 65535)); out_s.append(np.clip(before - c + inc, 0, 65535))
        lens.append(n); gm.append((we, be, base, inc, res))
    gm = np.array(gm, np.int64)
    offsets = np.zeros(len(lens) + 1, np.int64); np.cumsum(lens, out=offsets[1:])
    gs = GameStore(moves=np.concatenate(out_m), clock=np.concatenate(out_c).astype(np.uint16),
                   spent=np.concatenate(out_s).astype(np.uint16), offsets=offsets,
                   welo=gm[:, 0].astype(np.int16), belo=gm[:, 1].astype(np.int16), base=gm[:, 2].astype(np.int32),
                   inc=gm[:, 3].astype(np.int16), result=gm[:, 4].astype(np.int8))
    gs.save(a.out)
    print(f"{seen} games streamed, {gs.n_games} kept, {len(gs.moves)} plies -> {a.out}")


if __name__ == "__main__":
    main()
