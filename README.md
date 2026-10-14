# Cloverb

Cloverb is a human-like chess engine: a square-token transformer that predicts the move a human
of a given strength would play, superseding Chessformer / Maia-3. Its inputs are what a player
actually has at the board:

* the **player's Elo** (only the side to move; Maia-3 also needs the opponent's),
* the **time control**: base time in seconds and increment,
* the **time left** on the player's clock,
* the position and the **last 64 plies** of the game (Maia-3: 8 board snapshots).

Everything here (data processing, the Maia-3 benchmark, training, evaluation) was done on a single
4-core Intel Xeon (AMX) CPU without a GPU.

* Paper: [`paper/cloverb.pdf`](paper/cloverb.pdf)
* Maia-3 reproduction and critique: [`docs/critique.md`](docs/critique.md)
* Raw results: [`results/`](results)

## Maia-3 reproduced (ALLIE test set, Maia protocol, 100,016 positions)

| Checkpoint | Paper top-1 | Our top-1 | Our top-3 |
|---|---|---|---|
| Maia-3 79M | 57.1 | 56.82 | 83.51 |
| Maia-3 23M-ponder (reads the clock) | - | 56.53 | 83.32 |
| Maia-3 23M | 56.6 | 56.32 | 83.09 |
| Maia-3 5M | 55.4 | 55.27 | 82.33 |
| Maia-3 3M (ablation) | 54.8 | 54.58 | 81.77 |
| 5M, absolute PE / relative bias | 54.7 / 54.6 | 54.37 / 54.47 | 81.62 / 81.52 |
| 5M, no history / 32 positions | 54.0 / 55.4 | 53.79 / 55.15 | 81.34 / 82.26 |
| 5M as the released UCI engine runs it (history off) | - | **53.04** | 80.80 |

## Using the engine

Standalone C engine (no Python or PyTorch). It runs on any x86-64 or ARM CPU: `-march=native` selects AMX bf16 tiles
(Intel Sapphire Rapids and later), AVX-512, AVX2+FMA (e.g. AMD Ryzen) or portable C:

```bash
make cloverb-engine
./cloverb-engine checkpoints/cloverb.bin           # speaks UCI
./cloverb-engine checkpoints/cloverb.bin bench 1   # single-position latency
```

It has the UCI options `Elo`, `BaseTime`, `Temperature`, `MultiPV`, and `Threads`.

Python engine, with the same model plus predicted human think time:

```bash
python -m pip install torch numpy          # CPU wheels are enough
python -m cloverb.uci --model checkpoints/cloverb.pt
```

UCI options: `Elo` (the player Cloverb imitates), `BaseTime` (seconds; 0 = detect from the first
`go`), `Temperature` (0 = most likely human move, 1 = sample like a human), `MultiPV`, `HumanTime`
(wait as long as the predicted human think time). Increments and clocks come from `go wtime/btime
winc/binc`. The C core and the fused attention kernel are compiled on first use (`gcc`); on CPUs
without AVX-512 the model falls back to PyTorch attention.

## Layout

| Path | What |
|---|---|
| `csrc/cloverb_core.c` | chess rules, Lichess PGN parsing, batch building, FEN encoding (C, ctypes) |
| `csrc/attn64.c` | fused AVX-512 attention for 64-token boards (torch custom op in `cloverb/attn.py`) |
| `cloverb/model.py` | the Cloverb network |
| `cloverb/data.py`, `cloverb/train.py` | sampler, threaded batch loader, trainer (bf16/AMX, torch.compile) |
| `cloverb/teacher.py` | fast Maia-3 re-implementation (same weights) used as a distillation teacher |
| `cloverb/baseline.py` | Chessformer baseline built from the authors' model code |
| `csrc/cloverb_engine.c` | standalone UCI engine (AMX/AVX-512 inference, weights from `tools/export_engine.py`) |
| `cloverb/uci.py`, `cloverb/infer.py` | Python engine and inference API |
| `tools/` | store building, Maia-3 benchmark, evaluation, reports, tables |

## Reproducing

```bash
python tools/build_store.py data/store/2025-09-00000.npz <lichess parquet shard>   # ~2 min / 1.4M games
python tools/maia3_prep.py data/bench/allie2022.npz data/bench/allie2022_maia.npz
tools/run_maia3_suite.sh                                                         # Maia-3 benchmark
tools/launch.sh run.log -m cloverb.train --stores data/store/*.npz --out runs/x --compile
```
