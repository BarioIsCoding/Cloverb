#!/bin/bash
# Benchmark every released Maia-3 checkpoint on the 100k-position ALLIE subset (Maia protocol).
cd /home/user/cloverb
. /home/user/venv/bin/activate
S=/home/user/data/bench/allie2022.npz; P=/home/user/data/bench/allie2022_maia.npz; SUB=/home/user/data/bench/allie_sub100k.npy
O=results/maia3
for v in maia3-5m maia3-3m hist1 hist32 abspos relpos maia3-23m maia3-23m-ponder maia3-79m; do
  [ -f $O/allie100k_$v.json ] || python tools/bench_maia3.py --variant $v --store $S --prep $P --subset $SUB --out $O/allie100k_$v.npz | tail -1
done
[ -f $O/allie100k_maia3-5m_nohist.json ] || python tools/bench_maia3.py --variant maia3-5m --history-mode none --store $S --prep $P --subset $SUB --out $O/allie100k_maia3-5m_nohist.npz | tail -1
echo SUITE_DONE
