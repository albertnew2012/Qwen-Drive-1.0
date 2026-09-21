#!/usr/bin/env bash
# Lane type, fine-tuned. The base model defaults to "straight" (left recall 18%), which is
# a bias rather than blindness, so the training mix is balanced across the seven lane types.
# hires because the arrow is on the road surface and downscaling costs it detail.
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
CK=outputs/tlb/lora_lanetype_v1.pt
for a in 1 2 3 4 5 6; do
  E=""; [ -f $CK ] && E="--resume"
  $P local/tlb/train_lanetype_v1.py --train data/tlb/ft_lanetype_train.jsonl --cond hires \
     --rank 32 --alpha 64 --epochs 1 --accum 8 --lr 1e-4 --save-every 25 $E --save $CK \
     >> outputs/tlb/log_lanetype_ft.log 2>&1
  rc=$?; tail -1 outputs/tlb/log_lanetype_ft.log
  [ $rc -eq 0 ] && break
  echo "[lanetype crashed rc=$rc, resuming]"; sleep 20
done
echo "[lanetype eval] fine-tuned, hires"
$P local/tlb/eval_lanetype.py --question arrow --lora $CK --stride 3 \
   --out outputs/tlb/lanetype_ft_v1.json > outputs/tlb/log_lanetype_ft_eval.log 2>&1
grep -E "exact set|macro F1|^    left|^    straight|^    right|pred sets" outputs/tlb/log_lanetype_ft_eval.log
echo "LANETYPE_FT DONE"
