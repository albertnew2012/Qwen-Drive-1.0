#!/usr/bin/env bash
# v6: the only lever left untested. Resolution hurt (v4), capacity did nothing (v5), so
# hold the v3 recipe exactly and give it more passes over more data instead.
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
CK=outputs/tlb/lora_vqa_v6.pt
for a in 1 2 3 4 5 6 7 8; do
  E=""; [ -f $CK ] && E="--resume"
  $P local/tlb/train_vqa_v2.py --train data/tlb/ft_full_train_v4.jsonl --cond full \
     --rank 16 --alpha 32 --epochs 2 --accum 8 --lr 8e-5 --save-every 25 $E --save $CK \
     >> outputs/tlb/log_lora_v6.log 2>&1
  rc=$?; tail -1 outputs/tlb/log_lora_v6.log
  [ $rc -eq 0 ] && break
  echo "[v6 crashed rc=$rc, resuming]"; sleep 20
done
echo "[v6 eval] usable subset"
$P local/tlb/eval_vqa.py --src data/tlb/val.jsonl --cond full --question full \
   --lora $CK --out outputs/tlb/sub_vqa_v6.json > outputs/tlb/log_sub_vqa_v6.log 2>&1
grep -A 6 "===== full / full" outputs/tlb/log_sub_vqa_v6.log | head -8
echo "[v6 eval] full validation"
$P local/tlb/eval_vqa.py --src data/tlb/full_val.jsonl --cond full --question full \
   --lora $CK --out outputs/tlb/full_vqa_v6.json > outputs/tlb/log_full_vqa_v6.log 2>&1
grep -A 14 "===== full / full" outputs/tlb/log_full_vqa_v6.log | head -16
echo "VQA_V6 DONE"
