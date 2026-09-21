#!/usr/bin/env bash
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
while pgrep -f "[e]val_lane.py|[p]rompt_study.py" >/dev/null; do sleep 30; done
echo "[hint] position-hint on val, base model"
$P local/tlb/eval_vqa.py --cond full --question hint --out outputs/tlb/hint_base.json \
   > outputs/tlb/log_hint_base.log 2>&1
grep -A 12 "===== hint / full" outputs/tlb/log_hint_base.log | head -14
echo "[hint] position-hint on val, fine-tuned model"
$P local/tlb/eval_vqa.py --cond full --question hint --lora outputs/tlb/lora_vqa_nb.pt \
   --out outputs/tlb/hint_ft.json > outputs/tlb/log_hint_ft.log 2>&1
grep -A 12 "===== hint / full" outputs/tlb/log_hint_ft.log | head -14
echo "HINT2 QUEUE DONE"
