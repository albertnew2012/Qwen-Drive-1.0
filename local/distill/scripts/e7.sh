cd "$(dirname "$0")/../../.."
export PYTHONPATH=src:. CUDA_HOME=${CUDA_HOME:?set CUDA_HOME to a CUDA 12.8 toolkit}
export PATH="$PWD/.venv/bin:$PATH" TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0,1,2,3
echo "[$(date +%H:%M:%S)] E7 queued: waiting for E6 to finish (or fail)"
until grep -qE "E6_DONE|E6_LAUNCH_FAILED" outputs/logs/e6.log 2>/dev/null; do sleep 60; done
rm -rf outputs/distill/exp/e7 && mkdir -p outputs/distill/exp/e7
.venv/bin/python - <<'PY'
import torch, re
from pathlib import Path
e6ok = "E6_DONE" in Path("outputs/logs/e6.log").read_text()
src = "outputs/distill/exp/e6/student.pt" if e6ok else "outputs/distill/exp/e5c/student.pt"
ck = torch.load(src, map_location="cpu")
sd = ck["model"]
if not e6ok:  # E5c's head is 64x2; E7 uses 128x3, so it must be re-initialised
    for k in [k for k in sd if k.startswith("center.")]: del sd[k]
ck["model"] = sd; ck.pop("opt", None); ck.pop("ema", None)
ck["step"] = 0            # fresh one-cycle schedule over E7's own 20k steps
torch.save(ck, "outputs/distill/exp/e7/student.pt")
print(f"  E7 warm start from {src} (E6 {'ok' if e6ok else 'FAILED -> fallback'}), step reset to 0")
PY
echo "=== E7: E6 architecture + RECIPE (CBGS, EMA 0.999, camera flip), 20k steps ==="
.venv/bin/torchrun --nproc_per_node=4 --master_port=29623 \
  local/distill/train_student.py --det-objective center --det-head center --ref-points off \
  --center-hidden 128 --center-blocks 3 --center-min-radius 2 \
  --cbgs --ema 0.999 --cam-flip --steps 20000 --workers 8 --out outputs/distill/exp/e7 &
TPID=$!
sleep 200
if grep -qE "Traceback|Error" outputs/logs/e7.log || ! grep -qE "^  step " outputs/logs/e7.log \
   || ! grep -q "CBGS:" outputs/logs/e7.log || ! grep -q "EMA weights on" outputs/logs/e7.log; then
  echo "E7_LAUNCH_FAILED"; kill -9 $TPID 2>/dev/null; pkill -9 -f "master_port=2962[3]"; exit 1; fi
if grep -oE "cls [0-9.]+" outputs/logs/e7.log | tail -3 | grep -q "cls 0.0000"; then
  echo "E7_LAUNCH_FAILED: cls at 0.0000"; kill -9 $TPID 2>/dev/null; exit 1; fi
echo "[$(date +%H:%M:%S)] E7 200-second check passed: $(grep -oE 'cls [0-9.]+' outputs/logs/e7.log | head -4 | tr '\n' ' ')"
wait $TPID; echo "TRAIN_RC=$?"
source export_onnx/env_gpu.sh
for x in "e7::" "e7-nms2::--nms 2.0"; do t="${x%%::*}"; e="${x##*::}"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python local/distill/diagnose.py --ckpt outputs/distill/exp/e7/student.pt \
    --tag "$t" --note "E7: E6 arch + CBGS + EMA + cam-flip, 20k fresh-schedule steps (EMA weights)" --limit 250 $e; done
echo "E7_DONE"
