#!/bin/bash
# Final Cloverb run, resumable: label the remaining teacher-pool chunks (Maia-3-23M-ponder), wait for any
# diagnostics to finish, then train with the arguments in $ARGS_FILE (read at start, so the model
# configuration can be fixed after the diagnostics), snapshotting to the `checkpoints` branch every ~2 h.
# usage: tools/final_run2.sh NCHUNKS CHUNK_SIZE RUN_NAME ARGS_FILE [WAIT_PID]
NCH=$1; CH=$2; RUN=$3; ARGS_FILE=$4; WAIT_PID=$5
S=/home/user/data/store/merged4
P=/home/user/data/pool; mkdir -p $P
cd /home/user/cloverb
. /home/user/venv/bin/activate
export LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so.2 MALLOC_CONF="background_thread:true,dirty_decay_ms:-1,muzzy_decay_ms:-1"
if [ -n "$WAIT_PID" ]; then while kill -0 $WAIT_PID 2>/dev/null; do sleep 30; done; fi
POOLS=""
for i in $(seq 0 $((NCH - 1))); do
  f=$P/pool_$(printf %03d $i).npz
  [ -f $f ] || python -u tools/label_teacher.py --stores $S --out $f --n $CH --teacher maia3-23m-ponder --seed $((1000 + i)) \
      >> /home/user/runs/${RUN}_label.log 2>&1
  POOLS="$POOLS $f"
done
until grep -q DIAG2_DONE /home/user/runs/diag2_queue.log 2>/dev/null; do sleep 60; done
until [ -s $ARGS_FILE ]; do sleep 60; done
MARGS=$(cat $ARGS_FILE)
echo "training with: $MARGS" >> /home/user/runs/$RUN.log
python -u -m cloverb.train --stores $S --teacher-cache $POOLS --out /home/user/runs/$RUN $MARGS \
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
