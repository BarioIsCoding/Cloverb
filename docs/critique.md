# Chessformer / Maia-3: what we verified, and where it is sloppy

Sources read: the ICLR 2026 paper *Chessformer: A Unified Architecture for Chess Modeling*
(arXiv:2605.19091), the released repository `CSSLab/maia3` (inference only), the nine released
checkpoints on Hugging Face (`UofTCSSLab/Maia3-*`), the ALLIE test set (`yimingzhang/allie-data`),
and the ALLIE authors' independent Maia-3 scorer (`y0mingzhang/allie`, `src/allie/eval/maia3`).

## What holds up

* **The headline protocol is reproducible.** Dropping each game's first 10 plies and every ply from
  the first one whose mover has < 30 s left selects exactly **884,049** positions of the ALLIE test
  set, the number in the paper (`cloverb/gamestore.py: maia_protocol_mask`).
* **The accuracies reproduce.** On a fixed 100,016-position, whole-game sample of that protocol we
  measure (bf16 trunk, fp32 policy head, official model code; `results/maia3/`):

  | Variant | Paper | Ours (100k sample) | Top-3 (ours) |
  |---|---|---|---|
  | Maia-3 79M | 57.1 | 56.82 | 83.51 |
  | Maia-3 23M-ponder (clock) | - | 56.53 | 83.32 |
  | Maia-3 23M | 56.6 | 56.32 | 83.09 |
  | Maia-3 5M | 55.4 | 55.27 | 82.33 |
  | Maia-3 3M (ablation) | 54.8 | 54.58 | 81.77 |
  | 5M, absolute PE | 54.7 | 54.37 | 81.62 |
  | 5M, relative bias | 54.6 | 54.47 | 81.52 |
  | 5M, history n=0 (`hist1`) | 54.0 | 53.79 | 81.34 |
  | 5M, history n=31 (`hist32`) | 55.4 | 55.15 | 82.26 |

  Our sample is ~0.2 points harder than the full set across the board; orderings match.

## Sloppiness in the paper

1. **"We open-source all code and data."** The repository contains inference code only: no training
   loop, no data pipeline, no `ablate_size.sh` (which the model cards cite for the architecture
   hyper-parameters), no processed data. The models cannot be retrained from what was released.
2. **"FLOPs" are multiply-accumulates.** Table 7 gives 276M "FLOPs" for the 5M model; a forward
   pass of 4.9M parameters over 64 tokens is ~0.6 GFLOP. 276M is the MAC count. The same holds for
   the 427B quoted for AC-270M (20 x 270M x 79 tokens). Comparisons inside the paper are consistent,
   comparisons with FLOP-counted work are off by 2x.
3. **Parameter counts do not match the checkpoints.** Table 7: 4.91M (5M) and 2.98M (3M). The
   released checkpoints contain 5.23M and 3.15M unique parameters (incl. an undocumented head).
4. **An undocumented third head.** "All models we train have two output heads"; every checkpoint
   has a third ("ponder", time prediction) head, and one release (`Maia3-23M-ponder`) has a 30-way
   version that the released code cannot even load (size mismatch on `fc_ponder`).
5. **Value head width.** Section 3.3 says the value MLP has 128 hidden units; the code uses the
   model width (256/512/1024).
6. **Elo "soft embeddings" are a scalar in disguise.** The paper itself concedes that interpolating
   between two vectors "would achieve the same representational capacity" as a scalar; it then
   concatenates two such 128-d vectors to *every one of the 64 tokens*, so 256 of the 352 input
   dimensions (73%) carry two numbers, and the token projection recomputes the same affine function
   of Elo 64 times per position. The paper (gamma weights e_weak) and the code (the `low` embedding
   gets weight k/5000) also disagree on which vector is which.
7. **History in alternating orientation.** Each past board is mirrored to *its own* side to move
   (released `maia3.uci`, and the ALLIE authors' scorer). Consecutive frames are therefore flipped
   and colour-swapped relative to each other; to see "what moved" the network must first learn to
   undo a mirror between every pair of frames, through a single linear input projection.
8. **The n = 31 "no gain" conclusion is confounded.** History enters only as extra input channels
   squashed by one linear map (384 binary features per square for n = 31). "No significant gain from
   n = 7 to n = 31" shows that *this encoding* cannot use long history, not that long history is
   uninformative. (We confirm the measurement: hist32 55.15 vs 55.27.)
9. **The input is not the state the paper says it needs.** Section 4.3 lists castling rights,
   en passant, the halfmove clock and repetitions as part of the Markov state; the human models get
   none of them (the Leela variant gets all of them).
10. **The released engine runs without history by default.** `maia3-uci` fills every history slot
    with the current position unless `--use-uci-history` is passed. That input is in-distribution
    only for the 5% of training samples whose history was masked. Measured cost: **-2.2 points**
    (55.27 -> 53.04 top-1, 82.33 -> 80.80 top-3), worse than the model trained without any history
    (53.79).
11. **No clock.** The released models ignore the clock (the unreleased-in-code `Maia3-23M-ponder`
    checkpoint does read it and is +0.21 better even on the pressure-free protocol, 56.53 vs 56.32); the training data and the test protocol
    remove every position after a player first drops under 30 s, i.e. the most human part of a blitz
    game. The scaffolding that exists scales seconds by 1/100 without normalisation (classical base
    10800 s -> 108).
12. **FlashAttention citation.** "Formulating the interaction additively allows us to use existing
    memory-efficient attention kernels (Dao et al., 2022)". FlashAttention does not take arbitrary
    dense additive biases; the released code calls `nn.MultiheadAttention` with a float
    (B*H, 64, 64) mask, which takes the unfused path.
13. **Missing baseline for GAB.** GAB is compared with absolute embeddings and a *translation
    invariant* 15x15 relative bias, which cannot express edge effects (a knight on a1 has 2 moves).
    The obvious baseline, a static learned 64x64 bias per head and layer, is absent, so the gain
    cannot be attributed to GAB being *dynamic*. The paper's own Table 3 shows GAB maps correlate
    0.77 across positions.
14. **Training distribution != test distribution.** Training is balanced over 22 bins of *game
    average* rating (keeping at most 220 of every 20,000 games), while every headline number is on
    the natural distribution of the ALLIE test set, and per-mover behaviour is what is predicted.
15. **Uncontrolled comparisons.** Maia-3, ALLIE and Maia-2 differ in data period, data volume and
    compute; Table 1 attributes the gains to architecture. Confidence intervals are per-position
    binomial intervals although positions are clustered in games.
16. **Recipe.** Post-LN blocks (plus two more norms at the end) trained at lr 5e-5 for 1M steps with
    20 cosine restarts; post-LN is the usual reason a transformer needs such a small learning rate.

## Implementation inefficiencies in the released code

* Promotion logits are built by a 256-iteration Python loop of indexing and `cat`.
* The shared GAB template matrix is registered as a parameter of every layer and saved 9 times;
  the relative-bias ablation stores a constant 225x4096 one-hot buffer per layer (29 MB of 48 MB).
* bf16 inference: logits tie often enough that a strict "rank" count overstates top-1 by ~1 point
  (55.26 fp32 vs 56.44 naive bf16); the policy head has to stay in fp32 on CPU.

## What we learned, and what Cloverb does about it

| Maia-3 / Chessformer | Lesson | Cloverb |
|---|---|---|
| Squares as tokens + from-to attention policy head | Works; keep it | Same token layout; same 4352-move vocabulary (so Maia-3 can teach Cloverb) |
| GAB maps correlate 0.77 across positions | Most of the bias is static geometry | Per-layer, per-head static 64x64 bias + only 8 dynamic templates (GAB-lite), fused into the attention kernel |
| History = 8 alternating-orientation board snapshots through one linear map; "32 = 8" | The encoding, not chess, limits history | 64 plies encoded as *moves* scatter-added onto the squares they touched, one orientation, ~O(64 d) cost |
| Elo as two 128-d vectors concatenated to all 64 tokens | Skill is a scalar; let it act everywhere | Player Elo (hat-encoded) + clock features -> one conditioning vector -> adaLN-Zero in every block |
| No clock; time pressure removed from data and protocol | The clock changes human moves most when it is low | Base, increment and time left are inputs; time-pressure positions are trained on and evaluated |
| No castling / en passant / 50-move state | Missing state is free accuracy lost | Castling rights on rook squares, en-passant square, halfmove/fullmove counters |
| Post-LN, lr 5e-5, 1M steps on 8 A100s | Recipe, not architecture, sets the budget | Pre-norm + QK-norm, lr 1e-3, warmup + linear decay, EMA; trains on one 4-core CPU |
| Released engine without history by default (-2.2 points) | Defaults are part of the model | Cloverb's engine always uses the game's moves; history cannot be switched off |
| Policy logits in reduced precision | bf16 logits tie and inflate top-1 by ~1 point | Policy head in fp32 at inference / evaluation |
| `nn.MultiheadAttention` with float masks; Python promotion loop | Implementation matters on small sequences | Fused AVX-512 attention kernel (scores in L1), vectorised promotion logits, C data pipeline |
