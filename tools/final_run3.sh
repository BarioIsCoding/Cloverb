#!/bin/bash
# Supervised final run: train with the arguments in ARGS_FILE on the labelled teacher pools, resume from
# last.pt whenever the trainer dies (or exceeds MAX_GB of anonymous memory), snapshot to the
# `checkpoints` branch every ~2 h, and write runs/RUN.done only once the trainer reports completion.
# usage: tools/final_run3.sh RUN_NAME ARGS_FILE [MAX_GB]
RUN=$1; ARGS_FILE=$2; MAX_GB=${3:-12}
S=/home/user/data/store/merged4
OUT=/home/user/runs/$RUN
cd /home/user/cloverb
. /home/user/venv/bin/activate
# One jemalloc arena: with per-thread arenas and no decay, retained memory grows with every arena the
# compiled step touches (OOM after ~40 min); a single arena keeps it flat at the same speed.
export LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so.2 MALLOC_CONF="background_thread:true,dirty_decay_ms:-1,muzzy_decay_ms:-1,narenas:1"
POOLS=$(ls /home/user/data/pool/pool_*.npz)
MARGS=$(cat $ARGS_FILE)
n=0; tries=0
while [ ! -f $OUT/done ] && [ $tries -lt 30 ]; do
  tries=$((tries + 1))
  echo "$(date +%F_%T) start $tries: $MARGS" >> /home/user/runs/$RUN.log
  python -u -m cloverb.train --stores $S --teacher-cache $POOLS --out $OUT $MARGS \
     --compile --eval-every 2000 --ckpt-every 1000 --log-every 100 >> /home/user/runs/$RUN.log 2>&1 &
  TP=$!
  while kill -0 $TP 2>/dev/null; do
    sleep 60; n=$((n + 1))
    kb=$(awk '/RssAnon/{print $2}' /proc/$TP/status 2>/dev/null)
    if [ -n "$kb" ] && [ $kb -gt $((MAX_GB * 1048576)) ]; then
      echo "$(date +%F_%T) anon memory ${kb} kB > ${MAX_GB} GB: restarting from the last checkpoint" >> /home/user/runs/$RUN.log
      kill $TP; sleep 10; kill -9 $TP 2>/dev/null
    fi
    if [ $((n % 120)) -eq 0 ]; then
      tools/push_ckpt.sh $OUT $RUN >> /home/user/runs/${RUN}_push.log 2>&1
    fi
  done
  wait $TP; echo "$(date +%F_%T) trainer exited with $?" >> /home/user/runs/$RUN.log
done
tools/push_ckpt.sh $OUT $RUN >> /home/user/runs/${RUN}_push.log 2>&1
[ -f $OUT/done ] && echo FINAL_DONE > /home/user/runs/$RUN.done
