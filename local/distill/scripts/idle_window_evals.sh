#!/usr/bin/env bash
# Evaluations that need GPUs nobody else is using, run in the window between a lane's training
# end and its post-processing (which is single-GPU). Usage: idle_window_evals.sh "1 2 3" [CKPT] [ONNX]
#   GPU a: idle-GPU timings (untrained hist3 + 2x-head shape; deployment loop of the shipped ONNX)
#   GPU b: deliverable checkpoint on ALL 4,452 cached val-scene frames (tag <name>-valall)
#   GPU c: same with 2 m NMS (tag <name>-valall-nms2)
set -uo pipefail
GPUS=($1); CKPT=${2:-outputs/distill/exp/r2_long/snap_80000.pt}; ONNX=${3:-outputs/onnx/deliverable/student.onnx}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
export PYTHONPATH=src:. PATH="$PWD/.venv/bin:$PATH" TOKENIZERS_PARALLELISM=false; source export_onnx/env_gpu.sh
L=${DISTILL_CACHE:-/local/$USER/distill}; FR=$L/frames_real; TE=$L/teacher; NAME=$(basename $(dirname $CKPT))-$(basename $CKPT .pt | sed 's/snap_//; s/student_//')
mkdir -p outputs/logs/idle
( CUDA_VISIBLE_DEVICES=${GPUS[0]} .venv/bin/python local/distill/export_student.py --ckpt /nonexistent.pt --image-size 1152 640 --head-upsample 2 \
      --out outputs/onnx/shape_probe/hist3_head2.onnx --report outputs/distill/idle_hist3_head2_shape.json
  CUDA_VISIBLE_DEVICES=${GPUS[0]} .venv/bin/python local/distill/run_stateful.py --onnx $ONNX --gpu 0 --frames 40 ) > outputs/logs/idle/timing_gpu${GPUS[0]}.log 2>&1 &
( CUDA_VISIBLE_DEVICES=${GPUS[1]} .venv/bin/python local/distill/diagnose.py --ckpt $CKPT --tag ${NAME}-valall --tokens valall --limit 5000 \
      --frames $FR --teacher $TE --note "deliverable on every cached val-scene frame" ) > outputs/logs/idle/valall_gpu${GPUS[1]}.log 2>&1 &
if [ ${#GPUS[@]} -ge 3 ]; then
( CUDA_VISIBLE_DEVICES=${GPUS[2]} .venv/bin/python local/distill/diagnose.py --ckpt $CKPT --tag ${NAME}-valall-nms2 --tokens valall --limit 5000 --nms 2 \
      --frames $FR --teacher $TE --note "deliverable on every cached val-scene frame, 2 m NMS" ) > outputs/logs/idle/valall_nms2_gpu${GPUS[2]}.log 2>&1 &
fi
echo "idle-window evals launched on GPUs ${GPUS[*]} for $CKPT (tag $NAME); logs in outputs/logs/idle/"
