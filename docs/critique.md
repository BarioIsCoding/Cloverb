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
  | Maia-3 79M | 57.1 | see `results/maia3/allie100k_maia3-79m.json` | |
  | Maia-3 23M | 56.6 | see `results/maia3/allie100k_maia3-23m.json` | |
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
    only for the 5% of training samples whose history was masked. Measured cost: see
    `results/maia3/allie100k_maia3-5m_nohist.json`.
11. **No clock.** The released models ignore the clock; the training data and the test protocol
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
