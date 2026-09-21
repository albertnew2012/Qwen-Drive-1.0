#!/usr/bin/env bash
# v5 isolates the two changes v4 confounded. v4 raised BOTH resolution and LoRA rank and
# came out worse; this keeps the rank increase but returns to the pretraining pixel budget,
# so the comparison against v3 (rank 16, same budget) attributes the difference correctly.
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
CK=outputs/tlb/lora_vqa_v5.pt
for a in 1 2 3 4 5 6; do
  E=""; [ -f $CK ] && E="--resume"
  $P local/tlb/train_vqa_v2.py --train data/tlb/ft_full_train_v4.jsonl --cond full \
     --rank 32 --alpha 64 --epochs 1 --accum 8 --lr 1e-4 --save-every 25 $E --save $CK \
     >> outputs/tlb/log_lora_v5.log 2>&1
  rc=$?; tail -1 outputs/tlb/log_lora_v5.log
  [ $rc -eq 0 ] && break
  echo "[v5 crashed rc=$rc, resuming]"; sleep 20
done
echo "[v5 eval] usable subset"
$P local/tlb/eval_vqa.py --src data/tlb/val.jsonl --cond full --question full \
   --lora $CK --out outputs/tlb/sub_vqa_v5.json > outputs/tlb/log_sub_vqa_v5.log 2>&1
grep -A 8 "===== full / full" outputs/tlb/log_sub_vqa_v5.log | head -10
echo "[v5 eval] full validation split"
$P local/tlb/eval_vqa.py --src data/tlb/full_val.jsonl --cond full --question full \
   --lora $CK --out outputs/tlb/full_vqa_v5.json > outputs/tlb/log_full_vqa_v5.log 2>&1
grep -A 14 "===== full / full" outputs/tlb/log_full_vqa_v5.log | head -16
echo "VQA_V5 DONE"
