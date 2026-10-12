"""Paper figures from results/ and run logs. Palette: validated categorical slots, fixed order."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cloverb.gamestore import GameStore  # noqa: E402

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False})


def finish(ax, path, title):
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    fig = ax.figure
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    fig.savefig(Path(path).with_suffix(".pdf"))
    plt.close(fig)


def learning(runs, out, refs):
    """runs: list of (label, log.jsonl path); refs: {label: top1} horizontal references (gray)."""
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for i, (lab, p) in enumerate(runs):
        pts = [json.loads(l) for l in open(p) if '"eval_ema"' in l]
        x = [d["positions_seen"] / 1e6 for d in pts]
        y = [100 * d["eval_ema"]["top1"] for d in pts]
        ax.plot(x, y, color=SLOTS[i], lw=2, marker="o", ms=4, label=lab)
        ax.annotate(lab, (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points", color=INK, fontsize=8, va="center")
    for lab, v in refs.items():
        ax.axhline(100 * v, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.text(ax.get_xlim()[0], 100 * v + 0.15, lab, color=INK2, fontsize=7.5)
    ax.set_xlabel("training positions (millions)")
    ax.set_ylabel("top-1 move matching (%)")
    ax.legend(loc="lower right", fontsize=8)
    finish(ax, out, "Learning curves, ALLIE 10k sample")


def by_bins(store, results, out, key, edges, xlabel, title, min_n=300):
    gs = GameStore.load(store)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for i, (lab, f) in enumerate(results):
        z = np.load(f)
        idx, rank = z["idx"], z["rank"]
        g = np.searchsorted(gs.offsets, idx, side="right") - 1
        ply = idx - gs.offsets[g]
        if key == "elo":
            v = np.where(ply % 2 == 0, gs.welo[g], gs.belo[g])
        else:
            v = gs.clock[idx]
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (v >= lo) & (v < hi)
            if m.sum() >= min_n:
                xs.append((lo + hi) / 2 if key == "elo" else np.sqrt(max(lo, 1) * hi))
                ys.append(100 * np.mean(rank[m] == 0))
        ax.plot(xs, ys, color=SLOTS[i], lw=2, marker="o", ms=4, label=lab)
    if key != "elo":
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("top-1 move matching (%)")
    ax.legend(fontsize=8)
    finish(ax, out, title)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what")
    ap.add_argument("--spec", required=True, help="json spec")
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text())
    if a.what == "learning":
        learning(spec["runs"], spec["out"], spec.get("refs", {}))
    else:
        by_bins(spec["store"], spec["results"], spec["out"], spec["key"], spec["edges"], spec["xlabel"], spec["title"])
