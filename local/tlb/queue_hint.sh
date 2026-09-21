#!/usr/bin/env bash
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
while pgrep -f "[e]val_lane.py" >/dev/null; do sleep 30; done
echo "[hint 1/3] position-hint vs plain question on DEV (val untouched)"
$P local/tlb/prompt_study.py --cond full --prompts position_hint ego_jargon \
   --out-dir outputs/tlb/prompts_full > outputs/tlb/log_hint_dev.log 2>&1
grep -E "macroR " outputs/tlb/log_hint_dev.log
echo "[hint 2/3] position-hint on val, base model"
$P local/tlb/eval_vqa.py --cond full --question hint --out outputs/tlb/hint_base.json \
   > outputs/tlb/log_hint_base.log 2>&1
grep -A 12 "===== hint / full" outputs/tlb/log_hint_base.log | head -14
echo "[hint 3/3] position-hint on val, fine-tuned model"
$P local/tlb/eval_vqa.py --cond full --question hint --lora outputs/tlb/lora_vqa_nb.pt \
   --out outputs/tlb/hint_ft.json > outputs/tlb/log_hint_ft.log 2>&1
grep -A 12 "===== hint / full" outputs/tlb/log_hint_ft.log | head -14
echo "HINT QUEUE DONE"
