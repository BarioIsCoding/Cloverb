"""Generate paper/tables/*.tex from results/ (no numbers are typed by hand)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

PAPER = {"maia3-79m": 57.1, "maia3-23m": 56.6, "maia3-5m": 55.4, "maia3-3m": 54.8, "abspos": 54.7, "relpos": 54.6,
         "hist1": 54.0, "hist32": 55.4}
LABEL = {"maia3-79m": "Maia-3 79M", "maia3-23m": "Maia-3 23M", "maia3-23m-ponder": "Maia-3 23M-ponder (clock)",
         "maia3-5m": "Maia-3 5M", "maia3-3m": "Maia-3 3M (ablation)", "abspos": "5M, absolute PE",
         "relpos": "5M, relative bias", "hist1": "5M, no history", "hist32": "5M, 32 positions",
         "maia3-5m_nohist": "5M, UCI default (history off)"}


def j(p):
    return json.loads(Path(p).read_text())


def maia3_suite():
    rows = []
    for v in ["maia3-79m", "maia3-23m-ponder", "maia3-23m", "maia3-5m", "maia3-3m", "abspos", "relpos", "hist1", "hist32",
              "maia3-5m_nohist"]:
        f = R / "maia3" / f"allie100k_{v}.json"
        if not f.exists():
            continue
        d = j(f)
        paper = f"{PAPER[v]:.1f}" if v in PAPER else "--"
        rows.append(f"{LABEL[v]} & {d['params'] / 1e6:.1f}M & {paper} & {100 * d['top1']:.2f} & {100 * d['top3']:.2f} & {d['ce']:.3f} \\\\")
    tex = r"""\begin{table}[t]\centering\small
\caption{Released Maia-3 checkpoints on 100,016 positions of the ALLIE test set under Maia's protocol
(first 10 plies and time pressure removed). ``Paper'' is the accuracy reported on the full 884,049 positions.
95\% binomial half-width $\approx$0.31 points.}\label{tab:maia3}
\begin{tabular}{lrrrrr}\toprule
Checkpoint & Params & Paper top-1 & Top-1 & Top-3 & CE (nats)\\\midrule
""" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n"
    (OUT / "maia3_suite.tex").write_text(tex)
    a, b = j(R / "maia3/allie100k_maia3-5m.json"), j(R / "maia3/allie100k_maia3-5m_nohist.json")
    (OUT / "nohist_cost.tex").write_text(f"{100 * (a['top1'] - b['top1']):.1f}")


def run_curve(log):
    return {d["positions_seen"]: d["eval_ema"] for d in (json.loads(l) for l in open(log)) if "eval_ema" in d}


def arch_table(runs):
    """runs: [(label, log path, pos/s)] -> top-1 / CE at each evaluated sample count."""
    cols = sorted({n for _, p, _ in runs for n in run_curve(p)})
    head = " & ".join(f"{n / 1e6:.2f}M" for n in cols)
    rows = []
    for lab, p, speed in runs:
        c = run_curve(p)
        cells = " & ".join(f"{100 * c[n]['top1']:.1f} / {c[n]['ce']:.2f}" if n in c else "--" for n in cols)
        rows.append(f"{lab} & {speed} & {cells} \\\\")
    tex = (r"\begin{table}[t]\centering\small" "\n"
           r"\caption{Equal-data comparison on the ALLIE 10k sample (EMA weights): top-1 (\%) / cross-entropy after "
           r"$N$ training positions, same data stream and evaluation for all runs; throughput is end-to-end training "
           r"positions per second on the 4-core CPU.}\label{tab:arch}" "\n"
           r"\begin{tabular}{lr" + "c" * len(cols) + r"}\toprule" "\n"
           f"Run & pos/s & {head}\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")
    (OUT / "arch.tex").write_text(tex)


def main_table(sets, models):
    """sets: [(label, file stem)], models: [(label, results dir, file prefix)] -> top-1/top-3/CE per set."""
    rows = []
    for mlab, d, pre in models:
        cells = []
        for _, stem in sets:
            f = Path(d) / f"{pre}{stem}.json"
            if f.exists():
                x = j(f)
                cells.append(f"{100 * x['top1']:.2f} & {100 * x['top3']:.1f}")
            else:
                cells.append("-- & --")
        rows.append(f"{mlab} & " + " & ".join(cells) + " \\\\")
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{lab}}}" for lab, _ in sets)
    sub = " & ".join(["top-1 & top-3"] * len(sets))
    tex = (r"\begin{table*}[t]\centering\small" "\n"
           r"\caption{Move matching (\%) on every test set. ALLIE: 2022 blitz (Maia protocol = first 10 plies and "
           r"time pressure removed; all = every ply of the same games). Fresh: September 2026 games, unseen by every "
           r"model.}\label{tab:main}" "\n"
           r"\begin{tabular}{l" + "rr" * len(sets) + r"}\toprule" "\n"
           f" & {head}\\\\\n & {sub}\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table*}\n")
    (OUT / "main.tex").write_text(tex)


if __name__ == "__main__":
    maia3_suite()
    print("tables written to", OUT)
