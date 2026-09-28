#!/usr/bin/env bash
# e5 = e4 (hybrid objective) + the two fixes that were missing entirely:
#   1. lidar depth supervision on the lift-splat distribution
#   2. detection BEV at 1.02 m per token instead of 2.56 m (--pool 100)
# Warm-started from e4, so the backbone and BEV stack carry over; only `pos` is
# reinitialised (it is sized by pool) and the optimiser starts fresh because its Adam
# moments for `pos` still had the old shape.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=src:. CUDA_HOME=${CUDA_HOME:?set CUDA_HOME to a CUDA 12.8 toolkit}
export PATH="$PWD/.venv/bin:$PATH" TOKENIZERS_PARALLELISM=false
stamp() { date '+%H:%M:%S'; }

echo "[$(stamp)] e6 continuing to 220000 steps (pool 100, depth supervision on)"
.venv/bin/torchrun --nproc_per_node=4 --master_port=29655 \
    local/distill/train_student.py --det-objective hybrid --ref-points off \
    --pool 100 --depth-weight 1.0 --steps 220000 --workers 8 \
    --out outputs/distill/exp/e5
rc=$?
echo "[$(stamp)] training exited rc=$rc; evaluating"
for nms in 0 2.0; do
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python local/distill/eval_vs_gt.py \
      --ckpt outputs/distill/exp/e5/student.pt --limit 200 --nms $nms \
      --out outputs/distill/e6_gt_nms$nms.json 2>&1 | grep -E "TEACHER|STUDENT|DEGRAD"
done
CUDA_VISIBLE_DEVICES=0 .venv/bin/python local/distill/export_student.py \
    --out outputs/onnx/student_e6/student.onnx 2>&1 | grep -E "exported|WHOLE MODEL"
echo "[$(stamp)] E6 COMPLETE"
