#!/usr/bin/env bash
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
# The detector's misses are abstentions, not errors. Lower the objectness threshold to
# buy coverage and see what it costs in precision.
for T in 0.05 0.10 0.20; do
  $P local/tlb/eval_det.py --ckpt outputs/tlb/det_head.pt --thr $T \
     --out outputs/tlb/det_eval_t$T.json > outputs/tlb/log_det_eval_t$T.log 2>&1
  echo "--- thr=$T ---"
  grep -E "IoU 0.3|^  all " outputs/tlb/log_det_eval_t$T.log
done
echo "THR SWEEP DONE"
