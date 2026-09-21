#!/usr/bin/env bash
# LoRA fine-tune with crash recovery. GPU1 has thrown both an Xid 31 MMU fault and an
# autograd segfault today, so assume the run will die and make that cheap: checkpoint
# every 25 steps and resume in place.
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
CKPT=outputs/tlb/lora_vqa.pt

for attempt in 1 2 3 4 5 6; do
  EXTRA=""
  [ -f $CKPT ] && EXTRA="--resume"
  echo "[lora attempt $attempt] $EXTRA"
  $P local/tlb/train_vqa.py --train data/tlb/ft.jsonl --cond full --epochs 1 --accum 8 \
     --lr 1e-4 --balance --save-every 25 $EXTRA --save $CKPT \
     >> outputs/tlb/log_lora_train2.log 2>&1
  rc=$?
  tail -2 outputs/tlb/log_lora_train2.log
  if [ $rc -eq 0 ]; then echo "[lora finished on attempt $attempt]"; break; fi
  echo "[lora crashed rc=$rc -- resuming]"; sleep 20
done

echo "[lora eval] scoring the fine-tune on val"
$P local/tlb/eval_vqa.py --cond full --question assoc --lora $CKPT \
   --out outputs/tlb/ft_assoc_full.json > outputs/tlb/log_ft_eval.log 2>&1
grep -A 16 "===== assoc / full" outputs/tlb/log_ft_eval.log | head -18
echo "LORA QUEUE DONE"
