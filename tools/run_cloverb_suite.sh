#!/bin/bash
# Evaluate a Cloverb checkpoint on every benchmark set (+ eval-time ablations).
# usage: tools/run_cloverb_suite.sh CKPT TAG
CK=$1; TAG=$2
cd /home/user/cloverb
. /home/user/venv/bin/activate
export LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so.2
B=/home/user/data/bench; O=results/cloverb/$TAG; mkdir -p $O
ev() { # store subset name [extra args]
  local st=$1 sub=$2 name=$3; shift 3
  [ -f $O/$name.json ] || python tools/eval_cloverb.py --ckpt $CK --store $st --subset $sub --out $O/$name.npz --label "$TAG$LBL" "$@" | tail -1
}
LBL="" ev $B/allie2022.npz $B/allie_sub100k.npy allie100k
LBL="" ev $B/allie2022.npz $B/allie_sub_allplies.npy allieall
for s in fresh_blitz_maia fresh_bullet_all fresh_blitz_all fresh_rapid_all fresh_classical_all; do
  LBL="" ev $B/lichess2026-09.npz $B/$s.npy $s
done
LBL=" (clock shuffled)" ev $B/allie2022.npz $B/allie_sub_allplies.npy allieall_noclock --no-clock
LBL=" (history 8 plies)" ev $B/allie2022.npz $B/allie_sub100k.npy allie100k_hist8 --hist-limit 8
LBL=" (no history)" ev $B/allie2022.npz $B/allie_sub100k.npy allie100k_hist0 --hist-limit 0
LBL=" (clock shuffled)" ev $B/allie2022.npz $B/allie_sub100k.npy allie100k_noclock --no-clock
echo SUITE_DONE
