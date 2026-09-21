#!/usr/bin/env bash
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
echo "[gpu1 1/2] LoRA fine-tune, 1 epoch over the 211-segment ft split"
$P local/tlb/train_vqa.py --train data/tlb/ft.jsonl --cond full --epochs 1 --accum 8 \
   --lr 1e-4 --save outputs/tlb/lora_vqa.pt > outputs/tlb/log_lora_train.log 2>&1
tail -3 outputs/tlb/log_lora_train.log
echo "[gpu1 2/2] score the fine-tune on val (same question, same protocol as baseline)"
$P local/tlb/eval_vqa.py --cond full --question assoc --lora outputs/tlb/lora_vqa.pt \
   --out outputs/tlb/ft_assoc_full.json > outputs/tlb/log_ft_eval.log 2>&1
grep -A 16 "===== assoc / full" outputs/tlb/log_ft_eval.log | head -18
echo "GPU1 QUEUE DONE"
