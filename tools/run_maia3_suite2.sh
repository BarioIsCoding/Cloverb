#!/bin/bash
# Phase 2: Maia-3 on ALLIE all plies (same games as the 100k sample) and on fresh September-2026 games.
cd /home/user/cloverb
. /home/user/venv/bin/activate
B=/home/user/data/bench; O=results/maia3
run() { # variant store prep subset out
  [ -f $O/$5.json ] || python tools/bench_maia3.py --variant $1 --store $2 --prep $3 --subset $4 --out $O/$5.npz --fast | tail -1
}
for v in maia3-5m maia3-23m-ponder maia3-23m maia3-79m; do
  run $v $B/allie2022.npz $B/allie2022_maia.npz $B/allie_sub_allplies.npy allieall_$v
  for s in fresh_blitz_maia fresh_bullet_all fresh_blitz_all fresh_rapid_all fresh_classical_all; do
    run $v $B/lichess2026-09.npz $B/lichess2026-09_maia.npz $B/$s.npy ${s}_$v
  done
done
echo SUITE2_DONE
