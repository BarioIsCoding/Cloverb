# Cloverb

Cloverb is a human-like chess move-prediction engine: an optimized square-token transformer that
supersedes Chessformer / Maia-3. It is conditioned on the **player's Elo**, the **time control
(base seconds + increment)** and the **time left on the player's clock**, and it sees the **last
64 plies** of the game as moves (Maia-3 sees 8 board snapshots).

Work in progress. See `docs/` for the Maia-3 benchmark and critique, `results/` for raw numbers.

## Layout
- `csrc/cloverb_core.c` - chess rules, Lichess PGN parsing and batch building in C (ctypes)
- `cloverb/` - model, data pipeline, training, evaluation
- `tools/` - dataset building and Maia-3 benchmarking scripts
- `results/` - benchmark outputs (JSON)
