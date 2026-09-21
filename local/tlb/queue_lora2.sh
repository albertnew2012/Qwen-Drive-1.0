#!/usr/bin/env bash
# Second LoRA run, natural class distribution.
# The balanced run repeated the 103 yellow frames 5x and red-read-as-yellow rose 37 -> 60,
# which is 4.6% of val and the single reason the VQA path sits below 95%. Train on the
# real mix instead (2623 red / 1529 green / 103 yellow) and let the prior stand.
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
CKPT=outputs/tlb/lora_vqa_nb.pt
for attempt in 1 2 3 4 5 6; do
  EXTRA=""; [ -f $CKPT ] && EXTRA="--resume"
  echo "[lora-nb attempt $attempt] $EXTRA"
  $P local/tlb/train_vqa.py --train data/tlb/ft.jsonl --cond full --epochs 1 --accum 8 \
     --lr 1e-4 --save-every 25 --log-every 10 $EXTRA --save $CKPT \
     >> outputs/tlb/log_lora_nb.log 2>&1
  rc=$?; tail -2 outputs/tlb/log_lora_nb.log
  [ $rc -eq 0 ] && { echo "[lora-nb finished]"; break; }
  echo "[lora-nb crashed rc=$rc -- resuming]"; sleep 20
done
echo "[lora-nb eval]"
$P local/tlb/eval_vqa.py --cond full --question assoc --lora $CKPT \
   --out outputs/tlb/ft_nb_assoc_full.json > outputs/tlb/log_ft_nb_eval.log 2>&1
grep -A 16 "===== assoc / full" outputs/tlb/log_ft_nb_eval.log | head -18
echo "LORA-NB QUEUE DONE"
