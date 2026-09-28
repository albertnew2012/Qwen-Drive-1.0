#!/usr/bin/env bash
# Keep the GPUs earning after the orchestrator's last round finishes.
#
# DEFENSIVE, because two earlier versions misfired. Both were handed the pid of a
# transient wrapper that had already exited, so `kill -0` failed immediately, the script
# concluded "orchestrator done" and started a SECOND torchrun beside the live one --
# two jobs training into the same checkpoint. Hence the two guards below: refuse to
# start if the pid is already dead, and never start while anything still holds a GPU.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=src:. CUDA_HOME=${CUDA_HOME:?set CUDA_HOME to a CUDA 12.8 toolkit}
export PATH="$PWD/.venv/bin:$PATH" TOKENIZERS_PARALLELISM=false

ORCH=3115139
STEPS=78000

stamp() { date '+%H:%M:%S'; }

if ! kill -0 "$ORCH" 2>/dev/null; then
  echo "[$(stamp)] orchestrator $ORCH is not running -- refusing to start"
  exit 1
fi
echo "[$(stamp)] watching orchestrator $ORCH"
while kill -0 "$ORCH" 2>/dev/null; do sleep 60; done
echo "[$(stamp)] orchestrator exited"

# A rank can outlive its launcher, and starting beside one corrupts the checkpoint.
for _ in $(seq 1 120); do
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | grep -c . || true)
  [ "$n" -eq 0 ] && break
  echo "[$(stamp)] $n process(es) still on the GPUs; waiting"
  sleep 30
done

echo "[$(stamp)] continuing training to $STEPS steps"
.venv/bin/torchrun --nproc_per_node=4 --master_port=29544 \
    local/distill/train_student.py --steps $STEPS --workers 8
echo "[$(stamp)] training done; exporting and evaluating"

CUDA_VISIBLE_DEVICES=0 .venv/bin/python local/distill/export_student.py \
    --out outputs/onnx/student/student.onnx 2>&1 | grep -E "exported|WHOLE MODEL"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python local/distill/eval_student.py --limit 256
CUDA_VISIBLE_DEVICES=0 .venv/bin/python local/distill/eval_vs_gt.py \
    --ckpt outputs/distill/student/student.pt --limit 400 \
    --out outputs/distill/eval_vs_gt_final.json
echo "[$(stamp)] CONTINUATION COMPLETE"
