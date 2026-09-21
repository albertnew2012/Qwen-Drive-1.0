#!/usr/bin/env bash
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
# file-based wait: the k=10 result file is the sweep's last artifact. No pgrep, which is
# what deadlocked earlier when a wrapper matched its own command line.
while [ ! -f outputs/tlb/pipeline_soft_k10.json ]; do sleep 20; done
echo "[1/2] selector retrain: regularised, epoch chosen on overall answer accuracy"
$P local/tlb/train_selector.py --epochs 24 --dropout 0.25 --jitter 0.01 --feat-drop 0.1 \
   --select-on answer --save outputs/tlb/selector_v2.pt \
   --out outputs/tlb/selector_v2_val.json > outputs/tlb/log_selector_v2.log 2>&1
grep -E "^  ep|VAL|answer accuracy|discriminative|top-1" outputs/tlb/log_selector_v2.log | tail -6
echo "[2/2] pipeline with the regularised selector, soft vote k=3"
$P local/tlb/pipeline.py --selector outputs/tlb/selector_v2.pt \
   --colour outputs/tlb/colour_head_v2.pt --vote soft --topk 3 \
   --out outputs/tlb/pipeline_v3.json > outputs/tlb/log_pipeline_v3.log 2>&1
grep -E "^  all |^  discriminative|^  light <12px|^  light >=20px|selector picked|confusion" \
   outputs/tlb/log_pipeline_v3.log
echo "GPU0C QUEUE DONE"
