#!/bin/bash
# Experiment chain after run A: C (Cloverb + Maia-3-23M distillation), B (Chessformer-5M), Maia-3 phase 2.
S="/home/user/data/store/2025-09-00000.npz /home/user/data/store/2025-08-00010.npz /home/user/data/store/2025-08-00040.npz /home/user/data/store/2025-09-00030.npz"
wait_pid() { while kill -0 $1 2>/dev/null; do sleep 20; done; }
wait_pid $1
P=$(/home/user/cloverb/tools/launch.sh /home/user/runs/cmpC_distill.log -m cloverb.train --stores $S --out /home/user/runs/cmpC_distill \
  --d 256 --layers 8 --mlp 2 --templates 8 --steps 3000 --warmup 300 --decay-frac 0.25 --lr 1e-3 --eval-every 1000 \
  --ckpt-every 500 --log-every 50 --compile --teacher maia3-23m --distill-w 0.7)
wait_pid $P
P=$(/home/user/cloverb/tools/launch.sh /home/user/runs/cmpB_chessformer.log -m cloverb.train --stores $S --out /home/user/runs/cmpB_chessformer \
  --arch chessformer --cf-size 5m --steps 3000 --warmup 300 --decay-frac 0.25 --lr 5e-4 --eval-every 1000 \
  --ckpt-every 500 --log-every 50 --compile)
wait_pid $P
cd /home/user/cloverb && LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so.2 tools/run_maia3_suite2.sh > results/maia3/suite2.log 2>&1
echo CHAIN1_DONE > /home/user/runs/chain1.done
