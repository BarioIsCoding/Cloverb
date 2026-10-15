#!/bin/bash
# After the final run: export weights, evaluate everything, benchmark inference, build tables/figures/paper, push.
RUN=/home/user/runs/final
cd /home/user/cloverb
. /home/user/venv/bin/activate
until [ -f /home/user/runs/final.done ]; do sleep 60; done
mkdir -p checkpoints
python - <<'PY'
import json, torch
st = torch.load("/home/user/runs/final/last.pt", map_location="cpu", weights_only=False)
ev = [json.loads(l) for l in open("/home/user/runs/final/log.jsonl") if "eval_ema" in l][-1]
use_raw = ev["eval_raw"]["ce"] < ev["eval_ema"]["ce"]  # cross-entropy: far less noisy than top-1 on 10k positions
w = st["model"] if use_raw else st["ema"]
print("final weights:", "raw" if use_raw else "ema", ev)
torch.save(dict(ema=w, cfg=st["cfg"], step=st["step"], weights="raw" if use_raw else "ema"), "checkpoints/cloverb.pt")
PY
python tools/export_engine.py checkpoints/cloverb.pt checkpoints/cloverb.bin
cp $RUN/log.jsonl results/runs/final.jsonl; cp $RUN/config.json results/runs/final.config.json
tools/run_cloverb_suite.sh checkpoints/cloverb.pt final > results/cloverb_suite.log 2>&1
LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so.2 tools/run_maia3_suite2.sh > results/maia3/suite2.log 2>&1
python tools/bench_infer.py --cloverb checkpoints/cloverb.pt > results/infer_speed.log 2>&1
make cloverb-engine > /dev/null 2>&1
for t in 1 4; do ./cloverb-engine checkpoints/cloverb.bin bench $t; done > results/engine_speed.txt 2>&1
python tools/final_report.py > results/final_report.log 2>&1
python tools/make_tables.py > /dev/null 2>&1
(cd paper && pdflatex -interaction=nonstopmode cloverb.tex > /dev/null 2>&1; pdflatex -interaction=nonstopmode cloverb.tex > /dev/null 2>&1)
git add -A
git commit -q -m "Final Cloverb model, evaluation on all test sets, Maia-3 phase-2 benchmark, inference speed, paper

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GuuYF4BbwrDJDeSf6ggstk"
for i in 1 2 3 4; do git push -q origin main && break; sleep $((2**i)); done
echo POST_DONE > /home/user/runs/post.done
