#!/bin/bash
# Fine-tune the final model with a teacher that is reliable at every time control.
# Maia-3-23M-ponder, the clock-aware teacher of the main run, collapses at long base times on held-out
# training games (results/teacher_by_tc.json: 38% top-1 at 25+ min vs. 55.6% for Maia-3-23M), so:
#   1. drop pool entries with base >= 10 min (and the held-out validation games) from the ponder pools,
#   2. label 2.6M positions of games with base >= 10 min with Maia-3-23M (no clock input),
#   3. continue training from the final weights on both, 8k steps with a fresh short schedule.
# Idempotent: re-running skips finished steps and resumes training from last.pt.
set -e
cd /home/user/cloverb
. /home/user/venv/bin/activate
export LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so.2
S=/home/user/data/store/merged4; B=/home/user/data/bench; P=/home/user/data/pool; F=/home/user/data/pool_ft
R=/home/user/runs/ft; mkdir -p $R
PREC=$(grep -qw amx_bf16 /proc/cpuinfo || echo --fp32)
[ -f $R/before_tc.json ] || python tools/eval_cloverb.py --ckpt checkpoints/cloverb.pt --store $S \
    --subset $B/trainval_tc.npy --out $R/before_tc.npz --label "before fine-tune" | tail -1
[ -f $F/pool_005.npz ] || python tools/filter_pool.py --stores $S --pools $P/pool_00?.npz --out-dir $F \
    --max-base 599 --exclude $B/trainval_tc.npz
for i in 0 1; do
  [ -f $F/long_00$i.npz ] || python tools/label_teacher.py --stores $S --out $F/long_00$i.npz --n 1300000 \
      --teacher maia3-23m --seed $((5000 + i)) --min-base 600 --exclude $B/trainval_tc.npz
done
until [ -f $R/train.done ]; do
  python -u -m cloverb.train --stores $S --teacher-cache $F/pool_00?.npz $F/long_00?.npz --out $R \
     --d 256 --layers 10 --mlp 3 --heads 8 --templates 0 --geo-init --postln --cap-type \
     --lr 3e-4 --warmup 300 --decay-frac 0.5 --distill-w 1.0 --distill-w-pressure 0.4 --ema 0.999 \
     --steps 8000 --init checkpoints/cloverb.pt --init-step 0 $PREC \
     --compile --eval-every 2000 --ckpt-every 250 --log-every 100 && touch $R/train.done || sleep 30
done
echo FT_DONE
