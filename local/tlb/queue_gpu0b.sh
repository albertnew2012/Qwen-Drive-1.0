#!/usr/bin/env bash
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
for K in 3 6 10; do
  echo "[soft vote, topk=$K]"
  $P local/tlb/pipeline.py --colour outputs/tlb/colour_head_v2.pt --vote soft --topk $K \
     --out outputs/tlb/pipeline_soft_k$K.json > outputs/tlb/log_pipeline_soft_k$K.log 2>&1
  grep -E "^  all |^  discriminative|^  light <12px|selector picked|confusion" \
     outputs/tlb/log_pipeline_soft_k$K.log
done
echo "GPU0B QUEUE DONE"
