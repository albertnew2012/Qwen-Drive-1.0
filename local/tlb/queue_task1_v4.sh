#!/usr/bin/env bash
# Task 1 v3: same none-aware question, more positive evidence, to undo v2's over-abstention.
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
CK=outputs/tlb/lora_vqa_v3.pt
for a in 1 2 3 4 5; do
  E=""; [ -f $CK ] && E="--resume"
  $P local/tlb/train_vqa_v2.py --train data/tlb/ft_full_train_v3.jsonl --epochs 1 \
     --accum 8 --lr 1e-4 $E --save $CK >> outputs/tlb/log_lora_v3.log 2>&1
  rc=$?; tail -1 outputs/tlb/log_lora_v3.log
  [ $rc -eq 0 ] && break
  echo "[crashed rc=$rc, resuming]"; sleep 20
done
echo "[v3 eval] full validation split"
$P local/tlb/eval_vqa.py --src data/tlb/full_val.jsonl --cond full --question full \
   --lora $CK --out outputs/tlb/full_vqa_v3.json > outputs/tlb/log_full_vqa_v3.log 2>&1
grep -A 14 "===== full / full" outputs/tlb/log_full_vqa_v3.log | head -16
echo "[v3 eval] usable subset"
$P local/tlb/eval_vqa.py --src data/tlb/val.jsonl --cond full --question full \
   --lora $CK --out outputs/tlb/sub_vqa_v3.json > outputs/tlb/log_sub_vqa_v3.log 2>&1
grep -A 6 "===== full / full" outputs/tlb/log_sub_vqa_v3.log | head -8
echo "TASK1_V4 DONE"
