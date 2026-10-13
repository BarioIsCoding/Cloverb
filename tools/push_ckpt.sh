#!/bin/bash
# Snapshot a training run to the orphan `checkpoints` branch (amended + force-pushed: only the latest
# snapshot stays reachable, so the repo does not grow). usage: tools/push_ckpt.sh RUN_DIR NAME
RUN=$1; NAME=$2
[ -f "$1/last.pt" ] || { echo "no checkpoint in $1"; exit 0; }
cd /home/user/ckpt || exit 1
mkdir -p $NAME
. /home/user/venv/bin/activate
python - "$RUN" "$NAME" <<'PY'
import sys, torch, json, shutil
run, name = sys.argv[1], sys.argv[2]
st = torch.load(f"{run}/last.pt", map_location="cpu", weights_only=False)
torch.save(dict(ema=st["ema"], cfg=st["cfg"], step=st["step"], arch=st.get("arch", "cloverb")), f"{name}/ema.pt")
torch.save(dict(model=st["model"], step=st["step"]), f"{name}/model.pt")
shutil.copy(f"{run}/config.json", f"{name}/config.json")
shutil.copy(f"{run}/log.jsonl", f"{name}/log.jsonl")
print("step", st["step"])
PY
git add -A
if git rev-parse HEAD >/dev/null 2>&1; then git commit -q --amend -m "checkpoint snapshot"; else git commit -q -m "checkpoint snapshot"; fi
for i in 1 2 3 4; do git push -q -f origin checkpoints && break; sleep $((2**i)); done
