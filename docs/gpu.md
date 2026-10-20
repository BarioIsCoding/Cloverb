# Training Cloverb on an NVIDIA GPU

Everything in the paper ran on 4-core CPUs. The same code trains on CUDA with `--device cuda`:
attention then uses PyTorch's SDPA instead of the AVX-512 kernel, autocast uses bf16 tensor cores
(Ampere or newer, e.g. an RTX 3060), and batches are copied from pinned memory. The GPU path is
written to be equivalent to the tested CPU path but has not been run on a GPU by us yet; expect
roughly 10-15x the CPU's ~250 positions/s (bf16 on AMX) on an RTX 3060 (an estimate, not a measurement).

Windows: use WSL2 (Ubuntu) with the normal Windows NVIDIA driver; the data pipeline is C built with gcc.

## 1. Install

```bash
git clone https://github.com/BarioIsCoding/cloverb && cd cloverb
python3 -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # or the CUDA build pytorch.org lists for you
pip install -r requirements.txt
make                                   # C engine + the two shared libraries (gcc, -march=native)
export CLOVERB_DATA=$PWD/data CLOVERB_MODELS=$PWD/models
mkdir -p data/bench data/store data/raw data/pool models/maia3 && cp bench/* data/bench/
python -m pytest -q tests              # optional; the AVX-512 kernel tests skip on CPUs without it
```

## 2. Training games (Lichess, CC0)

The paper's run used four parquet shards from August/September 2025 (5.5M games, 372M plies). More
shards = more data; each shard is ~1 GB and builds in a couple of minutes.

```bash
H=https://huggingface.co/datasets/Lichess/standard-chess-games/resolve/main/data/year=2025
for s in month=08/train-00010-of-00066 month=08/train-00040-of-00066 \
         month=09/train-00000-of-00062 month=09/train-00030-of-00062; do
  curl -L -o data/raw/$(echo $s | tr '/=' '__').parquet "$H/$s.parquet"
done
for f in data/raw/*.parquet; do python tools/build_store.py data/store/$(basename $f .parquet).npz $f; done
python -c "import glob; from cloverb.data import concat_stores; concat_stores(sorted(glob.glob('data/store/*.npz'))).save_dir('data/store/merged')"
```

## 3. Teacher labels (Maia-3-23M-ponder)

```bash
python -c "from huggingface_hub import hf_hub_download as d; d('UofTCSSLab/Maia3-23M-ponder', 'maia3-23m-ponder.pt', local_dir='models/maia3')"
for i in 0 1 2 3 4 5; do
  python tools/label_teacher.py --device cuda --stores data/store/merged --teacher maia3-23m-ponder \
      --n 2000000 --seed $((1000 + i)) --out data/pool/pool_00$i.npz
done
```

The teacher is AGPL-3.0 (CSSLab). Using its predictions as training targets is covered in the
README (Third-party material).

## 4. Train

Fine-tune the paper's finished model (step 80k, shipped as `checkpoints/cloverb.pt`) for another 40k steps
with a fresh schedule (weights only; the optimiser restarts with a short warm-up, the learning rate decays
linearly over the last 40% as in the paper):

```bash
python -m cloverb.train --device cuda --compile --stores data/store/merged --teacher-cache data/pool/*.npz \
  --out runs/gpu --d 256 --layers 10 --mlp 3 --heads 8 --templates 0 --geo-init --postln --cap-type \
  --lr 3e-4 --warmup 300 --decay-frac 0.4 --distill-w 1.0 --distill-w-pressure 0.4 --ema 0.9998 \
  --steps 40000 --init checkpoints/cloverb.pt --init-step 0
```

or start a larger model from scratch (e.g. `--d 512 --layers 12 --heads 16`, about 37M parameters;
the C engine supports head dimensions that are multiples of 16 up to 64). Checkpoints, the EMA and
`log.jsonl` (evaluation on 10k ALLIE positions every 2000 steps) go to `--out`; the run resumes from
`last.pt` if restarted.

## 5. Evaluate and export

```bash
python tools/eval_cloverb.py --ckpt runs/gpu/ema.pt --store data/bench/allie2022.npz --subset data/bench/allie_sub100k.npy --out r.npz
python tools/export_engine.py runs/gpu/ema.pt cloverb.bin && ./cloverb-engine cloverb.bin bench 1
```
