#!/bin/bash
# After the final run: export weights, evaluate everything, benchmark inference, build tables/figures/paper, push.
# Kill and relaunch this script after editing it: bash reads it lazily while it waits.
RUN=/home/user/runs/final
B=/home/user/data/bench
cd /home/user/cloverb
. /home/user/venv/bin/activate
until [ -f /home/user/runs/final.done ]; do sleep 60; done
mkdir -p checkpoints results/cloverb/final
# Raw or EMA weights: chosen by cross-entropy on 10k Maia-protocol positions from ALLIE games that are in
# none of the test samples (the 10k sample monitored during training is part of the 100k test sample).
python - <<'PY'
import torch
st = torch.load("/home/user/runs/final/last.pt", map_location="cpu", weights_only=False)
for k in ("model", "ema"):
    torch.save(dict(ema=st[k], cfg=st["cfg"], step=st["step"]), f"/home/user/runs/final/sel_{k}.pt")
PY
for k in model ema; do
  [ -f $RUN/sel_$k.json ] || python tools/eval_cloverb.py --ckpt $RUN/sel_$k.pt --store $B/allie2022.npz \
      --subset $B/allie_sel10k.npy --out $RUN/sel_$k.npz --label "selection $k" | tail -1
done
python - <<'PY'
import json, torch
r = {k: json.load(open(f"/home/user/runs/final/sel_{k}.json")) for k in ("model", "ema")}
w = "model" if r["model"]["ce"] < r["ema"]["ce"] else "ema"
st = torch.load(f"/home/user/runs/final/sel_{w}.pt", map_location="cpu", weights_only=False)
name = "raw" if w == "model" else "ema"
torch.save(dict(ema=st["ema"], cfg=st["cfg"], step=st["step"], weights=name), "checkpoints/cloverb.pt")
sel = {("raw" if k == "model" else "ema"): {m: r[k][m] for m in ("top1", "top3", "ce")} for k in r}
json.dump(dict(weights=name, step=st["step"], selection_set="bench/allie_sel10k.npy", selection=sel),
          open("results/cloverb/final/weights.json", "w"), indent=2)
print("final weights:", name, sel)
PY
python tools/export_engine.py checkpoints/cloverb.pt checkpoints/cloverb.bin
cp $RUN/log.jsonl results/runs/final.jsonl; cp $RUN/config.json results/runs/final.config.json
tools/run_cloverb_suite.sh checkpoints/cloverb.pt final > results/cloverb_suite.log 2>&1
LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so.2 tools/run_maia3_suite2.sh > results/maia3/suite2.log 2>&1
python tools/bench_infer.py --cloverb checkpoints/cloverb.pt > results/infer_speed.log 2>&1
make cloverb-engine > results/engine_build.log 2>&1
for t in 1 4; do ./cloverb-engine checkpoints/cloverb.bin bench $t; done > results/engine_speed.txt 2>&1
python tools/final_report.py > results/final_report.log 2>&1
python tools/make_tables.py > /dev/null 2>&1
(cd paper && pdflatex -interaction=nonstopmode cloverb.tex > /dev/null 2>&1; pdflatex -interaction=nonstopmode cloverb.tex > /dev/null 2>&1)
PRE=$(git rev-parse HEAD)
git add -A
git commit -q -m "Final Cloverb model, evaluation on all test sets, Maia-3 phase-2 benchmark, inference speed, paper

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GuuYF4BbwrDJDeSf6ggstk"
for i in 1 2 3 4 5 6 7 8; do git push -q origin main && break; sleep $((30 * i)); done
if [ "$(git rev-parse HEAD)" != "$PRE" ] && [ "$(git rev-parse origin/main)" = "$(git rev-parse HEAD)" ]; then
  echo POST_DONE
else
  echo PUSH_OR_COMMIT_FAILED
fi > /home/user/runs/post.done
