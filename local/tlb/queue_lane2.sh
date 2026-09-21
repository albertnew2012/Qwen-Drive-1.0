#!/usr/bin/env bash
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
echo "[lane] same-direction wording, base model"
$P local/tlb/eval_lane.py --question same_dir --out outputs/tlb/lane_samedir.json \
   > outputs/tlb/log_lane_samedir.log 2>&1
grep -E "^  all |^  multi|predicted totals|gt totals|unparseable|sample" outputs/tlb/log_lane_samedir.log
echo "[lane] same wording, LoRA-tuned model (trained only on lights, so this tests transfer)"
$P local/tlb/eval_lane.py --question same_dir --lora outputs/tlb/lora_vqa_nb.pt \
   --out outputs/tlb/lane_samedir_ft.json > outputs/tlb/log_lane_samedir_ft.log 2>&1
grep -E "^  all |^  multi|predicted totals" outputs/tlb/log_lane_samedir_ft.log
echo "LANE2 QUEUE DONE"
