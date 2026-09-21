#!/usr/bin/env bash
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
for Q in arrow plain; do
  echo "[lanetype] question=$Q, base model"
  $P local/tlb/eval_lanetype.py --question $Q --out outputs/tlb/lanetype_$Q.json \
     > outputs/tlb/log_lanetype_$Q.log 2>&1
  grep -E "exact set|macro F1|^    left|^    straight|^    right|pred sets|sample" \
     outputs/tlb/log_lanetype_$Q.log
done
echo "LANETYPE QUEUE DONE"
