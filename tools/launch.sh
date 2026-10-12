#!/bin/bash
# Run a Python module with the CPU-tuned runtime: jemalloc (keeps freed pages, no page-fault storms
# on large tensors), 4 OpenMP threads pinned. usage: tools/launch.sh LOG -m cloverb.train ARGS...
LOG=$1; shift
cd /home/user/cloverb
. /home/user/venv/bin/activate
export LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so.2
export MALLOC_CONF="background_thread:true,dirty_decay_ms:-1,muzzy_decay_ms:-1"
export OMP_NUM_THREADS=4
nohup python -u "$@" > "$LOG" 2>&1 &
echo $!
