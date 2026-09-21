#!/usr/bin/env bash
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
CK=outputs/tlb/lora_vqa_v2.pt
for a in 1 2 3 4 5; do
  E=""; [ -f $CK ] && E="--resume"
  echo "[t1v2 attempt $a] $E"
  $P local/tlb/train_vqa_v2.py --epochs 1 --accum 8 --lr 1e-4 $E --save $CK \
     >> outputs/tlb/log_lora_v2.log 2>&1
  rc=$?; tail -2 outputs/tlb/log_lora_v2.log
  [ $rc -eq 0 ] && { echo "[done]"; break; }
  echo "[crashed rc=$rc, resuming]"; sleep 20
done
echo "[t1v2 eval] full validation split, all 6019 frames"
$P local/tlb/eval_vqa.py --src data/tlb/full_val.jsonl --cond full --question full \
   --lora $CK --out outputs/tlb/full_vqa_v2.json > outputs/tlb/log_full_vqa_v2.log 2>&1
grep -A 18 "===== full / full" outputs/tlb/log_full_vqa_v2.log | head -20
echo "[t1v2 eval] usable subset, for continuity with earlier numbers"
$P local/tlb/eval_vqa.py --src data/tlb/val.jsonl --cond full --question full \
   --lora $CK --out outputs/tlb/sub_vqa_v2.json > outputs/tlb/log_sub_vqa_v2.log 2>&1
grep -A 10 "===== full / full" outputs/tlb/log_sub_vqa_v2.log | head -12
echo "TASK1_V3 DONE"
