#!/bin/bash
# After tools/ft_longtc.sh: choose raw or EMA weights, export, evaluate everything, benchmark, build the paper, push.
# Idempotent (finished evaluations are skipped). Kill and relaunch after editing: bash reads scripts lazily.
RUN=/home/user/runs/ft
B=/home/user/data/bench
cd /home/user/cloverb
. /home/user/venv/bin/activate
export LD_PRELOAD=/lib/x86_64-linux-gnu/libjemalloc.so.2
until grep -q FT_DONE /home/user/runs/ft.log 2>/dev/null; do sleep 60; done
# the 80k-step model's results become the "before" rows
[ -d results/cloverb/v80k ] || git mv results/cloverb/final results/cloverb/v80k
mkdir -p results/cloverb/final
python - <<'PY'
import torch
st = torch.load("/home/user/runs/ft/last.pt", map_location="cpu", weights_only=False)
for k in ("model", "ema"):
    torch.save(dict(ema=st[k], cfg=st["cfg"], step=st["step"]), f"/home/user/runs/ft/sel_{k}.pt")
PY
# Raw or EMA: summed cross-entropy on two held-out sets, 10k ALLIE protocol positions (blitz) and the
# 480 held-out training games of every time control (bench/trainval_tc.npy); no test set is involved.
for k in model ema; do
  [ -f $RUN/sel_$k.json ] || python tools/eval_cloverb.py --ckpt $RUN/sel_$k.pt --store $B/allie2022.npz \
      --subset $B/allie_sel10k.npy --out $RUN/sel_$k.npz --label "selection $k" | tail -1
  [ -f $RUN/tc_$k.json ] || python tools/eval_cloverb.py --ckpt $RUN/sel_$k.pt --store /home/user/data/store/merged4 \
      --subset $B/trainval_tc.npy --out $RUN/tc_$k.npz --label "validation $k" | tail -1
done
python - <<'PY'
import json, torch
R = "/home/user/runs/ft"
sel = {k: json.load(open(f"{R}/sel_{k}.json")) for k in ("model", "ema")}
tc = {k: json.load(open(f"{R}/tc_{k}.json")) for k in ("model", "ema")}
w = min(("model", "ema"), key=lambda k: sel[k]["ce"] + tc[k]["ce"])
st = torch.load(f"{R}/sel_{w}.pt", map_location="cpu", weights_only=False)
name = "raw" if w == "model" else "ema"
torch.save(dict(ema=st["ema"], cfg=st["cfg"], step=st["step"], weights=name), "checkpoints/cloverb.pt")
pick = lambda d: {m: d[m] for m in ("top1", "top3", "ce")}  # noqa: E731
out = dict(weights=name, step=st["step"], base_model="80k-step main run (raw weights)",
           selection_sets=["bench/allie_sel10k.npy", "bench/trainval_tc.npy"],
           selection={("raw" if k == "model" else "ema"): dict(allie_sel10k=pick(sel[k]), trainval_tc=pick(tc[k]))
                      for k in sel})
json.dump(out, open("results/cloverb/final/weights.json", "w"), indent=2)
import shutil
for ext in ("json", "npz"):  # the chosen weights' validation scores, next to the 80k model's
    shutil.copy(f"{R}/tc_{w}.{ext}", f"results/cloverb/final/trainval_tc.{ext}")
    shutil.copy(f"{R}/before_tc.{ext}", f"results/cloverb/v80k/trainval_tc.{ext}")
print("final weights:", name, out["selection"])
PY
python tools/export_engine.py checkpoints/cloverb.pt checkpoints/cloverb.bin
cp $RUN/log.jsonl results/runs/ft_longtc.jsonl; cp $RUN/config.json results/runs/ft_longtc.config.json
tools/run_cloverb_suite.sh checkpoints/cloverb.pt final > results/cloverb_suite.log 2>&1
tools/run_maia3_suite2.sh > results/maia3/suite2_rest.log 2>&1
python tools/bench_infer.py --cloverb checkpoints/cloverb.pt > results/infer_speed.log 2>&1
make cloverb-engine > results/engine_build.log 2>&1
for t in 1 4; do ./cloverb-engine checkpoints/cloverb.bin bench $t; done > results/engine_speed.txt 2>&1
python tools/bench_attention.py --out results/attn_bench.json > results/attn_bench.log 2>&1
gcc -O3 -march=native -mno-amx-tile -mno-amx-bf16 -mno-amx-int8 -fopenmp -o /tmp/cloverb-engine-fp32 csrc/cloverb_engine.c -lm
python tools/check_engine.py --ckpt checkpoints/cloverb.pt --bin checkpoints/cloverb.bin --engine "AMX bf16 build=./cloverb-engine" \
    --engine "AVX-512 fp32 build=/tmp/cloverb-engine-fp32" --store $B/lichess2026-09.npz --n 500 --out results/engine_check.json > /dev/null 2>&1
python tools/final_report.py > results/final_report.log 2>&1
python tools/make_tables.py > /dev/null 2>&1
(cd paper && pdflatex -interaction=nonstopmode cloverb.tex > /dev/null 2>&1; pdflatex -interaction=nonstopmode cloverb.tex > /dev/null 2>&1)
PRE=$(git rev-parse HEAD)
git add -A
git commit -q -m "Cloverb after the long time-control fine-tune: weights, evaluation on all test sets, Maia-3 phase 2, benchmarks, paper

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01GuuYF4BbwrDJDeSf6ggstk"
for i in 1 2 3 4 5 6 7 8; do git push -q origin main && break; sleep $((30 * i)); done
if [ "$(git rev-parse HEAD)" != "$PRE" ] && [ "$(git rev-parse origin/main)" = "$(git rev-parse HEAD)" ]; then
  echo POST_DONE
else
  echo PUSH_OR_COMMIT_FAILED
fi > /home/user/runs/post_ft.done
