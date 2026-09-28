cd "$(dirname "$0")/../../.."
export PYTHONPATH=src:. CUDA_HOME=${CUDA_HOME:?set CUDA_HOME to a CUDA 12.8 toolkit}
export PATH="$PWD/.venv/bin:$PATH" TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0,1,2,3
echo "[$(date +%H:%M:%S)] E8 queued: waiting for E7 to finish (or fail)"
until grep -qE "E7_DONE|E7_LAUNCH_FAILED" outputs/logs/e7.log 2>/dev/null; do sleep 60; done
rm -rf outputs/distill/exp/e8 && mkdir -p outputs/distill/exp/e8
.venv/bin/python - <<'PY'
import json, torch
from pathlib import Path
def f1(r):
    v = r.get("student_vs_gt_best_thr")
    if isinstance(v, dict) and "f1" in v: return float(v["f1"])
    v = r.get("student_vs_gt")
    return float(v["f1"]) if isinstance(v, dict) and "f1" in v else -1.0
recs = json.load(open("outputs/distill/lab_notebook.json"))
best = {}
for r in recs:
    t = r.get("tag", "").split("-")[0]            # "e6-nms2" counts for e6
    if t in ("e6", "e7") and Path(f"outputs/distill/exp/{t}/student.pt").exists():
        best[t] = max(best.get(t, -1.0), f1(r))
src = max(best, key=best.get) if best else "e5c"
ck = torch.load(f"outputs/distill/exp/{src}/student.pt", map_location="cpu")
ema = ck.get("ema")
if ema:   # the EMA weights are the ones that score, so seed from them
    sd = {(k[len("module."):] if k.startswith("module.") else k): v
          for k, v in ema.items() if k != "n_averaged"}
else:
    sd = ck["model"]
if src == "e5c":   # 64x2 head -> 128x3: the head must be re-initialised
    for k in [k for k in sd if k.startswith("center.")]: del sd[k]
torch.save({"model": sd, "step": 0, "cfg": ck.get("cfg", {})}, "outputs/distill/exp/e8/student.pt")
print(f"  E8 warm start from {src} (best-thr F1 by run: {best}); {'EMA' if ema else 'raw'} weights; "
      f"step reset to 0; temporal_fuse initialised fresh (strict=False load)")
PY
echo "=== E8: E7 recipe + TEMPORAL fusion (previous keyframe BEV warped with ego motion), 20k steps ==="
.venv/bin/torchrun --nproc_per_node=4 --master_port=29624 \
  local/distill/train_student.py --det-objective center --det-head center --ref-points off \
  --center-hidden 128 --center-blocks 3 --center-min-radius 2 \
  --cbgs --ema 0.999 --cam-flip --temporal --steps 20000 --workers 8 --out outputs/distill/exp/e8 &
TPID=$!
sleep 600
# 51.75 M = E6/E7 architecture (49.10 M) + the 2.65 M temporal fuse: proves --temporal took effect
if grep -qE "Traceback|Error" outputs/logs/e8.log || ! grep -qE "^  step " outputs/logs/e8.log \
   || ! grep -q "CBGS:" outputs/logs/e8.log || ! grep -q "EMA weights on" outputs/logs/e8.log \
   || ! grep -q "student 51.75 M params" outputs/logs/e8.log; then
  echo "E8_LAUNCH_FAILED"; kill -9 $TPID 2>/dev/null; pkill -9 -f "master_port=2962[4]"; exit 1; fi
if grep -oE "cls [0-9.]+" outputs/logs/e8.log | tail -3 | grep -q "cls 0.0000"; then
  echo "E8_LAUNCH_FAILED: cls at 0.0000"; kill -9 $TPID 2>/dev/null; exit 1; fi
echo "[$(date +%H:%M:%S)] E8 600-second check passed: $(grep -oE 'cls [0-9.]+' outputs/logs/e8.log | head -4 | tr '\n' ' ')"
wait $TPID; echo "TRAIN_RC=$?"
source export_onnx/env_gpu.sh
for x in "e8::" "e8-nms2::--nms 2.0"; do t="${x%%::*}"; e="${x##*::}"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python local/distill/diagnose.py --ckpt outputs/distill/exp/e8/student.pt \
    --tag "$t" --note "E8: E7 recipe + temporal fusion (prev keyframe BEV), 20k fresh-schedule steps (EMA weights)" --limit 250 $e; done
echo "E8_DONE"
