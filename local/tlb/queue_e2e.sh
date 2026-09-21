#!/usr/bin/env bash
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
for T in 0.10 0.30; do
  echo "=== end-to-end, detector threshold $T ==="
  $P local/tlb/pipeline_e2e.py --thr $T --out outputs/tlb/e2e_val_t$T.json \
     > outputs/tlb/log_e2e_t$T.log 2>&1
  grep -E "^  all |^  discriminative|^  light |confusion" outputs/tlb/log_e2e_t$T.log
done
echo "E2E QUEUE DONE"
