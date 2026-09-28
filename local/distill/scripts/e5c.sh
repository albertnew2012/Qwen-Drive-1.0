cd "$(dirname "$0")/../../.."
export PYTHONPATH=src:. CUDA_HOME=${CUDA_HOME:?set CUDA_HOME to a CUDA 12.8 toolkit}
export PATH="$PWD/.venv/bin:$PATH" TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0,1,2,3
echo "=== E5c (Gaussian on integer cell, center head re-init, warm from e5 114.5k): dense CenterPoint head at the BEV's native 0.512 m/cell ==="
echo "    six DETR-style runs gave 0.0-2.1% F1. A linear probe on these same BEV"
echo "    features separates objects from empty cells at AUC 0.844, so the features"
echo "    carry the objects and the query head was the bottleneck."
.venv/bin/torchrun --nproc_per_node=4 --master_port=29621 \
  local/distill/train_student.py --det-objective center --det-head center \
  --ref-points off --steps 154500 --workers 8 --out outputs/distill/exp/e5c
echo "TRAIN_RC=$?"
source export_onnx/env_gpu.sh
for x in "e5c::" "e5c-nms2::--nms 2.0"; do
  t="${x%%::*}"; e="${x##*::}"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python local/distill/diagnose.py \
    --ckpt outputs/distill/exp/e5c/student.pt --tag "$t" \
    --note "dense CenterPoint head, warm-start BEV from distilled 78k, 20k steps" \
    --limit 250 $e
done
echo "E5C_DONE"
