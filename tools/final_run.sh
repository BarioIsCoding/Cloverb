#!/bin/bash
# Final Cloverb run: (1) label a pool of positions with the clock-aware Maia-3-23M-ponder teacher,
# in chunks so a crash loses at most one chunk; (2) train Cloverb for several epochs over the pool
# (soft teacher targets + the human move), snapshotting to the `checkpoints` branch every ~2 h.
# usage: tools/final_run.sh NCHUNKS CHUNK_SIZE RUN_NAME "MODEL ARGS" STEPS
NCH=$1; CH=$2; RUN=$3; MARGS=$4; STEPS=$5
S="/home/user/data/store/2025-09-00000.npz /home/user/data/store/2025-08-00010.npz /home/user/data/store/2025-08-00040.npz /home/user/data/store/2025-09-00030.npz"
P=/home/user/data/pool; mkdir -p $P
cd /home/user/cloverb
. /home/user/venv/bin/activate
export LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so.2 MALLOC_CONF="background_thread:true,dirty_decay_ms:-1,muzzy_decay_ms:-1"
POOLS=""
for i in $(seq 0 $((NCH - 1))); do
  f=$P/pool_$(printf %03d $i).npz
  [ -f $f ] || python -u tools/label_teacher.py --stores $S --out $f --n $CH --teacher maia3-23m-ponder --seed $((1000 + i)) \
      >> /home/user/runs/${RUN}_label.log 2>&1
  POOLS="$POOLS $f"
done
python -u -m cloverb.train --stores $S --teacher-cache $POOLS --out /home/user/runs/$RUN $MARGS --steps $STEPS \
   --compile --eval-every 2000 --ckpt-every 1000 --log-every 100 >> /home/user/runs/$RUN.log 2>&1 &
TP=$!
n=0
while kill -0 $TP 2>/dev/null; do
  sleep 60; n=$((n + 1))
  if [ $((n % 120)) -eq 0 ] && [ -f /home/user/runs/$RUN/last.pt ]; then
    tools/push_ckpt.sh /home/user/runs/$RUN $RUN >> /home/user/runs/${RUN}_push.log 2>&1
  fi
done
tools/push_ckpt.sh /home/user/runs/$RUN $RUN >> /home/user/runs/${RUN}_push.log 2>&1
echo FINAL_DONE > /home/user/runs/$RUN.done
