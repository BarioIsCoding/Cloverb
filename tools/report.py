"""Aggregate per-position results (bench_maia3.py / eval_cloverb.py outputs) into tables.

Every result file is an .npz with idx (global ply index into a store), ce and rank, plus a .json.
Breakdowns: mover Elo band, time control, time left, opening plies. Paired differences against a
reference model use a game-clustered bootstrap (2000 draws), since positions of a game are correlated.

usage: python tools/report.py --store S.npz --results A.npz B.npz ... [--ref A.npz] [--md out.md]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloverb.data import tc_class  # noqa: E402
from cloverb.gamestore import GameStore  # noqa: E402

ELO_BANDS = [(0, 1000), (1000, 1400), (1400, 1800), (1800, 2200), (2200, 9999)]
CLOCK_BANDS = [(0, 10), (10, 30), (30, 60), (60, 1e9)]


def boot_ci(games, x, reps=2000, seed=7):
    uniq, inv = np.unique(games, return_inverse=True)
    sums = np.bincount(inv, weights=x, minlength=len(uniq)); ns = np.bincount(inv, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(uniq), (reps, len(uniq)))
    b = sums[pick].sum(1) / ns[pick].sum(1)
    return np.percentile(b, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    gs = GameStore.load(a.store)
    res = {}
    for f in a.results:
        z = np.load(f)
        info = json.loads(Path(f).with_suffix(".json").read_text())
        name = info.get("label") or info["variant"] + ("-nohist" if info.get("history_mode") == "none" else "")
        res[name] = (z["idx"], z["ce"], z["rank"], f)
    names = list(res)
    idx0 = res[names[0]][0]
    for n in names:
        assert np.array_equal(res[n][0], idx0), f"{n} scored different positions"
    g = np.searchsorted(gs.offsets, idx0, side="right") - 1
    ply = idx0 - gs.offsets[g]
    white = ply % 2 == 0
    elo = np.where(white, gs.welo[g], gs.belo[g])
    clock = gs.clock[idx0]
    tc = tc_class(gs.base, gs.inc)[g]
    lines = [f"### {a.title}" if a.title else "", "",
             f"{len(idx0):,} positions from {len(np.unique(g)):,} games.", "",
             "| Model | Top-1 (%) | Top-3 (%) | CE (nats) |" + (" Δ top-1 vs ref [95% CI] |" if a.ref else ""),
             "|---|---|---|---|" + ("---|" if a.ref else "")]
    ref = None
    if a.ref:
        ref = [n for n in names if res[n][3] == a.ref][0]
    for n in names:
        _, ce, rank, _ = res[n]
        row = f"| {n} | {100 * np.mean(rank == 0):.2f} | {100 * np.mean(rank < 3):.2f} | {ce.mean():.4f} |"
        if ref:
            if n == ref:
                row += " - |"
            else:
                d = 100 * ((rank == 0).astype(float) - (res[ref][2] == 0))
                lo, hi = boot_ci(g, d)
                row += f" {d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] |"
        lines.append(row)
    lines += ["", "Top-1 (%) by mover Elo:", "",
              "| Model | " + " | ".join(f"{lo}-{hi}" if hi < 9999 else f">={lo}" for lo, hi in ELO_BANDS) + " |",
              "|---|" + "---|" * len(ELO_BANDS)]
    for n in names:
        r = res[n][2]
        lines.append(f"| {n} | " + " | ".join(
            f"{100 * np.mean(r[(elo >= lo) & (elo < hi)] == 0):.1f}" if ((elo >= lo) & (elo < hi)).any() else "-"
            for lo, hi in ELO_BANDS) + " |")
    if (clock < 30).any():
        lines += ["", "Top-1 (%) by time left on the mover's clock (s):", "",
                  "| Model | " + " | ".join(f"{lo}-{hi}" if hi < 1e9 else f">={lo}" for lo, hi in CLOCK_BANDS) + " |",
                  "|---|" + "---|" * len(CLOCK_BANDS)]
        for n in names:
            r = res[n][2]
            lines.append(f"| {n} | " + " | ".join(
                f"{100 * np.mean(r[(clock >= lo) & (clock < hi)] == 0):.1f}" if ((clock >= lo) & (clock < hi)).any() else "-"
                for lo, hi in CLOCK_BANDS) + " |")
        cnt = [int(((clock >= lo) & (clock < hi)).sum()) for lo, hi in CLOCK_BANDS]
        lines.append("| positions | " + " | ".join(f"{c:,}" for c in cnt) + " |")
    if len(np.unique(tc)) > 1:
        lines += ["", "Top-1 (%) by time control:", "", "| Model | bullet | blitz | rapid | classical |", "|---|---|---|---|---|"]
        for n in names:
            r = res[n][2]
            lines.append(f"| {n} | " + " | ".join(f"{100 * np.mean(r[tc == c] == 0):.1f}" if (tc == c).any() else "-"
                                                  for c in range(4)) + " |")
    md = "\n".join(lines)
    print(md)
    if a.md:
        Path(a.md).write_text(md + "\n")


if __name__ == "__main__":
    main()
