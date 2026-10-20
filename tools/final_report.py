"""Generate every final table, figure and summary from results/ (run after the final evaluation).

Writes paper/tables/{main,fresh,ablation,timing,segments,tcfix,infer,abstract_results}.tex, paper/figures/*.png|pdf,
results/paired.json and results/summary.md. The prose discussing them (paper/sections/final_text.tex) is written by hand.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import make_figures as mf  # noqa: E402

R = ROOT / "results"
TAB = ROOT / "paper" / "tables"
CLOV = R / "cloverb" / "final"
BENCH = Path("/home/user/data/bench")
MAIA = ["maia3-79m", "maia3-23m-ponder", "maia3-23m", "maia3-5m"]
V80K = R / "cloverb" / "v80k"   # the 80k-step model before the long time-control fine-tune
LAB = {"maia3-79m": "Maia-3 79M", "maia3-23m-ponder": "Maia-3 23M-ponder", "maia3-23m": "Maia-3 23M",
       "maia3-5m": "Maia-3 5M", "cloverb": "\\textbf{Cloverb} (9.2M)", "cloverb80k": "\\quad before the fix (80k steps)"}
CLOVERBS = ["cloverb", "cloverb80k"]


def path(model, stem, ext):
    d = {"cloverb": CLOV, "cloverb80k": V80K}.get(model)
    return (d / f"{stem}.{ext}") if d else (R / "maia3" / f"{stem}_{model}.{ext}")


def res(model, stem):
    p = path(model, stem, "json")
    return json.loads(p.read_text()) if p.exists() else None


def npz(model, stem):
    p = path(model, stem, "npz")
    return np.load(p) if p.exists() else None


def cell(d, k="top1", pct=True, nd=2):
    if d is None:
        return "--"
    return f"{100 * d[k]:.{nd}f}" if pct else f"{d[k]:.3f}"


def main_table():
    rows = []
    for m in MAIA + CLOVERBS:
        a, b = res(m, "allie100k"), res(m, "allieall")
        if a is None and b is None:
            continue
        rows.append(f"{LAB[m]} & {cell(a)} & {cell(a, 'top3', nd=1)} & {cell(a, 'ce', False)} & "
                    f"{cell(b)} & {cell(b, 'top3', nd=1)} & {cell(b, 'ce', False)} \\\\")
    tex = (r"\begin{table}[t]\centering\small" "\n"
           r"\caption{ALLIE 2022 blitz test set. Left: Maia's protocol (100,016 positions; first 10 plies and time "
           r"pressure removed). Right: every ply of the same 2,101 games (141,390 positions), including openings and "
           r"time trouble. Maia-3 sees both ratings; Cloverb only the mover's.}\label{tab:main}" "\n"
           r"\begin{tabular}{lrrrrrr}\toprule" "\n"
           r" & \multicolumn{3}{c}{Maia protocol} & \multicolumn{3}{c}{All plies}\\" "\n"
           r"Model & Top-1 & Top-3 & CE & Top-1 & Top-3 & CE\\\midrule" "\n" + "\n".join(rows) +
           "\n\\bottomrule\\end{tabular}\\end{table}\n")
    (TAB / "main.tex").write_text(tex)


def n_games(store, subset):
    off = np.load(store)["offsets"]
    return len(np.unique(np.searchsorted(off, np.load(subset), side="right") - 1))


def fresh_table():
    sets = [("fresh_blitz_maia", "Blitz (Maia prot.)"), ("fresh_bullet_all", "Bullet"), ("fresh_blitz_all", "Blitz"),
            ("fresh_rapid_all", "Rapid"), ("fresh_classical_all", "Classical")]
    ng = {s: n_games(BENCH / "lichess2026-09.npz", BENCH / f"{s}.npy") for s, _ in sets}
    lo, hi = min(ng[s] for s, _ in sets[1:]), max(ng[s] for s, _ in sets[1:])
    rows = []
    for m in MAIA + CLOVERBS:
        if res(m, sets[0][0]) is None:
            continue
        rows.append(f"{LAB[m]} & " + " & ".join(f"{cell(res(m, s))} / {cell(res(m, s), 'top3', nd=1)}" for s, _ in sets) + " \\\\")
    tex = (r"\begin{table*}[t]\centering\small" "\n"
           r"\caption{Fresh games from September 2026 (unseen by every model), top-1 / top-3 (\%). All-plies sets "
           r"are every ply of whole games, about 25k positions per set from " f"{lo}--{hi}" r" games each; the "
           r"Maia-protocol set has 50k positions from " f"{ng['fresh_blitz_maia']:,}" r" games. Positions of one game are "
           r"correlated, so differences below about one point are within noise.}\label{tab:fresh}" "\n"
           r"\begin{tabular}{l" + "c" * len(sets) + r"}\toprule" "\n"
           "Model & " + " & ".join(l for _, l in sets) + r"\\\midrule" "\n" + "\n".join(rows) +
           "\n\\bottomrule\\end{tabular}\\end{table*}\n")
    (TAB / "fresh.tex").write_text(tex)


def ablation_table():
    rows = []
    for stem, lab in [("allie100k", "Cloverb (full inputs)"), ("allie100k_hist8", "\\quad history cut to 8 plies"),
                      ("allie100k_hist0", "\\quad no history"), ("allie100k_noclock", "\\quad clock shuffled"),
                      ("allieall", "Cloverb, all plies"), ("allieall_noclock", "\\quad clock shuffled")]:
        d = res("cloverb", stem)
        rows.append(f"{lab} & {cell(d)} & {cell(d, 'top3', nd=1)} & {cell(d, 'ce', False)} \\\\")
    tex = (r"\begin{table}[t]\centering\small" "\n"
           r"\caption{What the final model uses, measured by removing inputs at evaluation time on the ALLIE sets "
           r"(the clock is replaced by another position's fraction of base time left).}\label{tab:ablation}" "\n"
           r"\begin{tabular}{lrrr}\toprule" "\n" r"Evaluation & Top-1 & Top-3 & CE\\\midrule" "\n" + "\n".join(rows) +
           "\n\\bottomrule\\end{tabular}\\end{table}\n")
    (TAB / "ablation.tex").write_text(tex)


def timing_table():
    sets = [("allieall", "ALLIE 2022 blitz"), ("fresh_bullet_all", "Fresh bullet"), ("fresh_blitz_all", "Fresh blitz"),
            ("fresh_rapid_all", "Fresh rapid"), ("fresh_classical_all", "Fresh classical")]
    rows = []
    for stem, lab in sets:
        d = res("cloverb", stem)
        if d is None or "time_mae_log" not in d:
            continue
        rows.append(f"{lab} & {d['time_mae_log']:.3f} & {d['time_mae_log_const']:.3f} & {d['time_mae_log_rule']:.3f} & "
                    f"{d['time_corr_log']:.2f} & {100 * d['time_within_2x']:.1f} & {100 * d['value_acc']:.1f} & {d['value_ce']:.3f} \\\\")
    if not rows:
        return
    tex = (r"\begin{table*}[t]\centering\small" "\n"
           r"\caption{Think time and outcome, all plies with a known time spent. Mean absolute error of "
           r"$\log(1+\text{seconds})$ for Cloverb's median predicted think time, for the best constant (the "
           r"test set's own median, an oracle) and for the rule of thumb ``1/40 of the clock plus the increment''; "
           r"correlation of predicted and actual log time; share of moves predicted within a factor of two. "
           r"Right: accuracy and cross-entropy of the win/draw/loss head against the game result.}\label{tab:timing}" "\n"
           r"\begin{tabular}{lrrrrrrr}\toprule" "\n"
           r" & \multicolumn{3}{c}{Log-time MAE} & & & \multicolumn{2}{c}{Outcome}\\" "\n"
           r"Test set & Cloverb & Constant & Rule & Corr. & $\le2\times$ (\%) & Acc. (\%) & CE\\\midrule" "\n"
           + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table*}\n")
    (TAB / "timing.tex").write_text(tex)


STORE = {"allie": BENCH / "allie2022.npz", "fresh": BENCH / "lichess2026-09.npz"}
_stores = {}


def store_of(stem):
    k = "allie" if stem.startswith("allie") else "fresh"
    if k not in _stores:
        z = np.load(STORE[k])
        _stores[k] = {f: z[f] for f in ("offsets", "clock")}
    return _stores[k]


def clustered(d, game):
    """Mean of per-position values d and its 95% half-width with positions clustered by game."""
    _, g = np.unique(game, return_inverse=True)
    s, n = np.bincount(g, d), np.bincount(g)
    m = s.sum() / n.sum()
    k = len(n)
    return m, 1.96 * np.sqrt(k / (k - 1) * np.sum((s - m * n) ** 2)) / n.sum()


def paired():
    """Cloverb minus each Maia-3 model in top-1 on every test set, paired on positions, clustered by game."""
    out = {}
    for stem in ["allie100k", "allieall", "fresh_blitz_maia", "fresh_bullet_all", "fresh_blitz_all", "fresh_rapid_all",
                 "fresh_classical_all"]:
        c = npz("cloverb", stem)
        if c is None:
            continue
        game = np.searchsorted(store_of(stem)["offsets"], c["idx"], side="right") - 1
        for m in MAIA + ["cloverb80k"]:
            z = npz(m, stem)
            if z is None:
                continue
            assert (z["idx"] == c["idx"]).all(), (m, stem)
            d, hw = clustered((c["rank"] == 0).astype(float) - (z["rank"] == 0), game)
            out.setdefault(stem, {})[m] = {"top1_diff": 100 * d, "ci95": 100 * hw, "positions": int(len(game))}
    (R / "paired.json").write_text(json.dumps(out, indent=2) + "\n")


def segments(stem, model):
    """Top-1 (%) on openings (ply < 10), normal play and time trouble (< 30 s on the mover's clock)."""
    z = npz(model, stem)
    if z is None:
        return None
    st = store_of(stem)
    idx = z["idx"]
    ply = idx - st["offsets"][np.searchsorted(st["offsets"], idx, side="right") - 1]
    clk = st["clock"][idx]
    seg = {"open": ply < 10, "normal": (ply >= 10) & (clk >= 30), "pressure": (ply >= 10) & (clk < 30)}
    return {k: (100 * float(np.mean(z["rank"][v] == 0)), int(v.sum())) for k, v in seg.items()}


def segment_table():
    fresh = ["fresh_bullet_all", "fresh_blitz_all", "fresh_rapid_all", "fresh_classical_all"]
    rows, ns = [], {}
    for m in MAIA + CLOVERBS:
        a = segments("allieall", m)
        fs = [segments(s, m) for s in fresh]
        if a is None:
            continue
        cells = [f"{a[k][0]:.2f}" for k in ("open", "normal", "pressure")]
        ns["allie"] = [a[k][1] for k in ("open", "normal", "pressure")]
        if all(fs):  # pooled over the four fresh all-plies sets
            tot = {k: sum(f[k][1] for f in fs) for k in ("open", "normal", "pressure")}
            cells += [f"{sum(f[k][0] * f[k][1] for f in fs) / tot[k]:.2f}" for k in ("open", "normal", "pressure")]
            ns["fresh"] = [tot[k] for k in ("open", "normal", "pressure")]
        else:
            cells += ["--"] * 3
        rows.append(f"{LAB[m]} & " + " & ".join(cells) + " \\\\")
    if not rows:
        return
    n = lambda k: ", ".join(f"{x:,}" for x in ns.get(k, []))  # noqa: E731
    tex = (r"\begin{table}[t]\centering\small" "\n"
           r"\caption{Top-1 (\%) by game phase on all plies: openings (first 10 plies), normal play, and time "
           r"trouble (later plies with under 30\,s on the mover's clock, which Maia's protocol and training data "
           r"exclude). ALLIE: 2022 blitz (" + n("allie") + r" positions); fresh: the four September 2026 all-plies "
           r"sets pooled (" + n("fresh") + r").}\label{tab:segments}" "\n"
           r"\begin{tabular}{lrrrrrr}\toprule" "\n"
           r" & \multicolumn{3}{c}{ALLIE, all plies} & \multicolumn{3}{c}{Fresh 2026, all plies}\\" "\n"
           r"Model & Open. & Normal & $<$30\,s & Open. & Normal & $<$30\,s\\\midrule" "\n"
           + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")
    (TAB / "segments.tex").write_text(tex)


def tc_table():
    """Top-1 by base time on 480 held-out training games: the two teachers and Cloverb before/after the fix."""
    t = json.loads((R / "teacher_by_tc.json").read_text())
    S = Path("/home/user/data/store/merged4")
    off, base = np.load(S / "offsets.npy"), np.load(S / "base.npy")
    sys.path.insert(0, str(ROOT / "tools"))
    from teacher_by_tc import BUCKETS, bucket_of
    names = [b[0] for b in BUCKETS]
    rows = [f"{LAB[m]} (teacher) & " + " & ".join(f"{t['teachers'][m][n]:.1f}" for n in names) + " \\\\"
            for m in ("maia3-23m-ponder", "maia3-23m") if m in t["teachers"]]
    for m, lab in (("cloverb80k", "Cloverb, 80k steps (before)"), ("cloverb", "\\textbf{Cloverb}, fine-tuned (after)")):
        z = npz(m, "trainval_tc")
        if z is None:
            continue
        b = bucket_of(base[np.searchsorted(off, z["idx"], side="right") - 1])
        rows.append(f"{lab} & " + " & ".join(f"{100 * np.mean(z['rank'][b == i] == 0):.1f}" for i in range(len(names)))
                    + " \\\\")
    tex = (r"\begin{table}[t]\centering\small" "\n"
           r"\caption{Top-1 (\%) by base time on 480 held-out games of the training months (120 per column, all "
           r"plies; " + ", ".join(f"{v:,}" for v in t["positions"].values()) + r" positions), kept out of the fine-tuning pools (the original pools hold 2,280 of their positions among "
           r"12M entries). "
           r"The clock-aware teacher collapses at long time controls; the fix relabels games with a base time of 10 "
           r"minutes or more with Maia-3-23M.}\label{tab:tcfix}" "\n"
           r"\begin{tabular}{lrrrr}\toprule" "\n"
           r"Model & 3--5 min & 10 min & 15--20 min & 25+ min\\\midrule" "\n" + "\n".join(rows) +
           "\n\\bottomrule\\end{tabular}\\end{table}\n")
    (TAB / "tcfix.tex").write_text(tex)


def infer_table():
    rows = []
    p = R / "infer_speed.json"
    if p.exists():
        for k, v in json.loads(p.read_text()).items():
            name, prec, batch, thr = [x.strip() for x in k.split("|")]
            rows.append(f"{name} & {prec} & {batch.replace('batch ', '')} & {thr.replace(' thr', '')} & {v['ms']:.2f} & {v['pos_per_s']:.0f} \\\\")
    p = R / "engine_speed.txt"
    if p.exists():  # e.g. "AMX bf16: 1.234 ms/position (1 thread), top move e2e4 0.123"
        for line in p.read_text().splitlines():
            m = re.match(r"^(.+?): ([\d.]+) ms/position \((\d+) threads?\)", line)
            if m:
                ms = float(m.group(2))
                rows.append(f"Cloverb C engine$^*$ & {m.group(1)} & 1 & {m.group(3)} & {ms:.2f} & {1000 / ms:.0f} \\\\")
    if not rows:
        return
    cpu = "the same 4-core CPU"
    p = R / "infer_cpu.json"
    if p.exists():
        c = json.loads(p.read_text())
        cpu = f"a {c['cores']}-core Intel Xeon " + ("with AMX" if c["amx"] else "without AMX (AVX-512, no bf16 units)")
    tex = (r"\begin{table}[t]\centering\small" "\n"
           r"\caption{Inference on " + cpu + r" (PyTorch rows: model forward only; latency = batch 1, 1 thread). "
           r"$^*$Standalone C engine, including input encoding and the softmax over legal moves.}\label{tab:infer}" "\n"
           r"\begin{tabular}{llrrrr}\toprule" "\n" r"Model & Precision & Batch & Threads & ms & pos/s\\\midrule" "\n"
           + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")
    (TAB / "infer.tex").write_text(tex)


def figures():
    allie = "/home/user/data/bench/allie2022.npz"
    fresh = "/home/user/data/bench/lichess2026-09.npz"
    m = [(LAB[x].replace("\\textbf{", "").replace("}", ""), str(R / "maia3" / f"allie100k_{x}.npz")) for x in ["maia3-79m", "maia3-23m", "maia3-5m"]]
    m.insert(0, ("Cloverb", str(CLOV / "allie100k.npz")))
    c = [("Cloverb", str(CLOV / "allieall.npz")), ("Maia-3 79M", str(R / "maia3/allieall_maia3-79m.npz")),
         ("Maia-3 23M-ponder (clock)", str(R / "maia3/allieall_maia3-23m-ponder.npz")),
         ("Maia-3 5M", str(R / "maia3/allieall_maia3-5m.npz"))]
    mf.by_bins(allie, c, str(ROOT / "paper/figures/by_clock.png"), "clock", [1, 5, 10, 20, 30, 45, 60, 90, 120, 180, 300],
               "time left on the mover's clock (s)", "Top-1 by time left, ALLIE (all plies)")
    mf.by_bins(allie, m, str(ROOT / "paper/figures/by_elo.png"), "elo", list(range(600, 3000, 200)), "mover rating",
               "Top-1 by mover rating, ALLIE (Maia protocol)")
    log = R / "runs" / "final.jsonl"
    if log.exists():
        ref = json.loads((R / "maia3/allie10k_reference.json").read_text())
        mf.learning([("Cloverb final run", str(log))], str(ROOT / "paper/figures/final_curve.png"),
                    {"Maia-3 79M": ref["maia3-79m"]["top1"], "Maia-3 5M": ref["maia3-5m"]["top1"]})


def summary():
    lines = ["# Results summary", "", "| Test set | Cloverb top-1 | Cloverb top-3 | Maia-3 79M top-1 | Maia-3 5M top-1 |", "|---|---|---|---|---|"]
    for stem, lab in [("allie100k", "ALLIE, Maia protocol"), ("allieall", "ALLIE, all plies"), ("fresh_blitz_maia", "Fresh blitz, Maia protocol"),
                      ("fresh_bullet_all", "Fresh bullet"), ("fresh_blitz_all", "Fresh blitz"), ("fresh_rapid_all", "Fresh rapid"),
                      ("fresh_classical_all", "Fresh classical")]:
        c, a, b = res("cloverb", stem), res("maia3-79m", stem), res("maia3-5m", stem)
        f = lambda d, k="top1": f"{100 * d[k]:.2f}" if d else "-"  # noqa: E731
        lines.append(f"| {lab} | {f(c)} | {f(c, 'top3')} | {f(a)} | {f(b)} |")
    (R / "summary.md").write_text("\n".join(lines) + "\n")
    c, a, b = res("cloverb", "allie100k"), res("maia3-79m", "allie100k"), res("maia3-5m", "allie100k")
    if c and a and b:
        (TAB / "abstract_results.tex").write_text(
            f"On the ALLIE benchmark under Maia's protocol the final 9.2M-parameter Cloverb reaches "
            f"{100 * c['top1']:.1f}\\% top-1 and {100 * c['top3']:.1f}\\% top-3 move matching "
            f"(Maia-3-5M: {100 * b['top1']:.1f}\\%, Maia-3-79M: {100 * a['top1']:.1f}\\% on the same positions).")
    print("\n".join(lines))


if __name__ == "__main__":
    for f in (main_table, fresh_table, ablation_table, timing_table, segment_table, tc_table, paired, infer_table, summary,
              figures):
        try:
            f()
        except Exception as e:  # keep going: a missing result should not block the rest
            print(f"{f.__name__} failed: {e}")
