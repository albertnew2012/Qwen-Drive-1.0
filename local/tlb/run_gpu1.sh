#!/usr/bin/env bash
# GPU1 work queue, strictly serial. No pgrep waiting: each step simply follows the last,
# which is what the earlier chained waiters failed to do (their own command lines
# contained the pattern they were waiting on, so they matched themselves and hung).
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python

echo "[1/3] the four lost prompts, incl. both association-free controls"
$P local/tlb/prompt_study.py --cond hires --prompts salient any_light ego_reversed mcq \
   > outputs/tlb/log_prompts_c.log 2>&1
grep -E "macroR " outputs/tlb/log_prompts_c.log

echo "[2/3] val ROI cache"
$P local/tlb/cache_roi.py --split val > outputs/tlb/log_roi_val.log 2>&1
tail -2 outputs/tlb/log_roi_val.log

echo "[3/3] colour head retrain, epoch chosen on held-out TRAIN segments"
$P local/tlb/train_colour.py --balance --epochs 14 \
   --save outputs/tlb/colour_head_clean.pt > outputs/tlb/log_colour_clean.log 2>&1
tail -3 outputs/tlb/log_colour_clean.log
echo "GPU1 QUEUE DONE"
