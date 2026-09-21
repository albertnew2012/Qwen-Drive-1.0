#!/usr/bin/env bash
# Task 1 v4: native-resolution fine-tune. The VLM's failures are on segments the colour
# head reads perfectly, so the information is in the pixels -- the model just cannot
# resolve a ~12 px lamp after the default 0.8x downscale. hires keeps 1600x900.
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
CK=outputs/tlb/lora_vqa_v4.pt
for a in 1 2 3 4 5 6; do
  E=""; [ -f $CK ] && E="--resume"
  $P local/tlb/train_vqa_v2.py --train data/tlb/ft_full_train_v4.jsonl --cond hires \
     --rank 32 --alpha 64 --epochs 1 --accum 8 --lr 1e-4 --save-every 25 $E --save $CK \
     >> outputs/tlb/log_lora_v4.log 2>&1
  rc=$?; tail -1 outputs/tlb/log_lora_v4.log
  [ $rc -eq 0 ] && break
  echo "[v4 crashed rc=$rc, resuming]"; sleep 20
done
echo "[v4 eval] usable subset at hires"
$P local/tlb/eval_vqa.py --src data/tlb/val.jsonl --cond hires --question full \
   --lora $CK --out outputs/tlb/sub_vqa_v4.json > outputs/tlb/log_sub_vqa_v4.log 2>&1
grep -A 8 "===== full / hires" outputs/tlb/log_sub_vqa_v4.log | head -10
echo "[v4 eval] full validation split at hires"
$P local/tlb/eval_vqa.py --src data/tlb/full_val.jsonl --cond hires --question full \
   --lora $CK --out outputs/tlb/full_vqa_v4.json > outputs/tlb/log_full_vqa_v4.log 2>&1
grep -A 14 "===== full / hires" outputs/tlb/log_full_vqa_v4.log | head -16
echo "VQA_V4 DONE"
