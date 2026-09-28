cd "$(dirname "$0")/../../.."
export PYTHONPATH=src:. CUDA_HOME=${CUDA_HOME:?set CUDA_HOME to a CUDA 12.8 toolkit}
export PATH="$PWD/.venv/bin:$PATH" TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0,1,2,3
rm -rf outputs/distill/exp/e6 && mkdir -p outputs/distill/exp/e6
.venv/bin/python - <<'PY'
import torch
ck = torch.load("outputs/distill/exp/e5c/student.pt", map_location="cpu")
sd = ck["model"]
for k in [k for k in sd if k.startswith("center.")]: del sd[k]     # 64x2 head -> 128x3
torch.save({"model": sd, "step": 0, "cfg": ck.get("cfg", {})}, "outputs/distill/exp/e6/student.pt")
print(f"  E6 warm start from E5c final (step {ck.get('step')}), center.* dropped, step reset to 0: fresh 20k one-cycle")
PY
echo "=== E6 (relaunched $(date +%H:%M)): center head 128x3, min radius 2, warm from E5c trunk, 20k fresh-schedule steps ==="
.venv/bin/torchrun --nproc_per_node=4 --master_port=29622 \
  local/distill/train_student.py --det-objective center --det-head center --ref-points off \
  --center-hidden 128 --center-blocks 3 --center-min-radius 2 \
  --steps 20000 --workers 8 --out outputs/distill/exp/e6 &
TPID=$!
sleep 200
if grep -qE "Traceback|Error" outputs/logs/e6.log || ! grep -qE "^  step " outputs/logs/e6.log \
   || ! grep -q "student 49.10 M params" outputs/logs/e6.log; then
  echo "E6_LAUNCH_FAILED"; kill -9 $TPID 2>/dev/null; pkill -9 -f "master_port=2962[2]"; exit 1; fi
if grep -oE "cls [0-9.]+" outputs/logs/e6.log | tail -3 | grep -q "cls 0.0000"; then
  echo "E6_LAUNCH_FAILED: cls at 0.0000"; kill -9 $TPID 2>/dev/null; exit 1; fi
echo "[$(date +%H:%M:%S)] E6 3-minute check passed: $(grep -oE 'cls [0-9.]+' outputs/logs/e6.log | head -4 | tr '\n' ' ')"
wait $TPID; echo "TRAIN_RC=$?"
source export_onnx/env_gpu.sh
for x in "e6::" "e6-nms2::--nms 2.0"; do t="${x%%::*}"; e="${x##*::}"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python local/distill/diagnose.py --ckpt outputs/distill/exp/e6/student.pt \
    --tag "$t" --note "E6: center head 128x3 min-radius 2, warm from E5c final trunk, 20k fresh-schedule steps" --limit 250 $e; done
echo "E6_DONE"
