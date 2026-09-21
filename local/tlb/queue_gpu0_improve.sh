#!/usr/bin/env bash
# GPU0 improvement pass: verify the unfrozen 3D head did not cost 2D accuracy, then
# attack the real Task 2 bottleneck -- the selector, whose top-1 pick is the ego's light
# only 86.7% of the time while colour given a box is 99.85%.
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python

echo "[1/3] CONTROL: does unfreezing cost 2D accuracy? (colour acc must hold at ~97.4%)"
$P local/tlb/pipeline_3d_v1.py --head3d outputs/tlb/det_head3d_v2_unfrozen.pt \
   --out outputs/tlb/pipeline3d_val_unfrozen.json > outputs/tlb/log_pipeline3d_unf.log 2>&1
grep -E "colour, all|range error|height error" outputs/tlb/log_pipeline3d_unf.log

echo "[2/3] selector seeds for an ensemble (single seed overfits by epoch 1)"
for S in 1 2 3 4; do
  $P local/tlb/train_selector_v2.py --epochs 24 --seed $S --dropout 0.15 --jitter 0.01 \
     --feat-drop 0.1 --select-on answer --save outputs/tlb/selector_s$S.pt \
     > outputs/tlb/log_selector_s$S.log 2>&1
  printf "  seed %s: %s\n" $S "$(grep -A 3 '===== VAL' outputs/tlb/log_selector_s$S.log | grep -E 'answer accuracy' | sed 's/^ *//')"
done
echo "[3/3] seeds trained"
echo "GPU0_IMPROVE DONE"
