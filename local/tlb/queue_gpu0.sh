#!/usr/bin/env bash
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
echo "[gpu0 1/3] colour head, epoch chosen on governing-light accuracy (held-out train segs)"
$P local/tlb/train_colour.py --balance --epochs 14 --save outputs/tlb/colour_head_v2.pt \
   > outputs/tlb/log_colour_v2.log 2>&1
grep -E "^  ep|best SELECTION" outputs/tlb/log_colour_v2.log | tail -5
echo "[gpu0 2/3] oracle-box check of the new head"
$P local/tlb/eval_colour_oracle.py --ckpt outputs/tlb/colour_head_v2.pt \
   --out outputs/tlb/colour_oracle_v2.json > outputs/tlb/log_colour_oracle_v2.log 2>&1
grep -E "argmax|confusion" outputs/tlb/log_colour_oracle_v2.log
echo "[gpu0 3/3] end-to-end pipeline with the new head"
$P local/tlb/pipeline.py --colour outputs/tlb/colour_head_v2.pt \
   --out outputs/tlb/pipeline_val_v2.json > outputs/tlb/log_pipeline_v2.log 2>&1
grep -A 10 "===== PIPELINE" outputs/tlb/log_pipeline_v2.log
echo "GPU0 QUEUE DONE"
