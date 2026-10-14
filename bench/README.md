# Benchmark sets

Game stores (`cloverb/gamestore.py` format) and the position subsets every number in the paper uses.
Point `CLOVERB_DATA` at a directory whose `bench/` holds these files (or copy them to `$CLOVERB_DATA/bench/`).

| File | Content |
|---|---|
| `allie2022.npz` | ALLIE 2022 Lichess blitz test set (18,239 games), from the [ALLIE repository](https://github.com/y0mingzhang/allie) (MIT), converted with `GameStore.from_allie_jsonl` |
| `allie_sub100k.npy` | 100,016 positions (2,101 whole games) of Maia's protocol: plies 11+, stop at the first ply with < 30 s on the mover's clock |
| `allie_sub10k.npy` | 10k-position subset of the above used for evaluation during training |
| `allie_sub_allplies.npy` | every ply (141,390) of the same 2,101 games, openings and time pressure included |
| `lichess2026-09.npz` | 120k games streamed from the September 2026 Lichess dump (CC0), unseen by every model |
| `fresh_*_all.npy` | 25k positions each of bullet / blitz / rapid / classical from those games, all plies |
| `fresh_blitz_maia.npy` | 50k blitz positions of the fresh games under Maia's protocol |

Each `.npy` holds global ply indices into its store (`store.offsets[g] + ply`).
