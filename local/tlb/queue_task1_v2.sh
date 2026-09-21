#!/usr/bin/env bash
# Task 1 explicit-form evaluations, after the GPU1 threshold work finishes.
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
while [ ! -f outputs/tlb/full_pipeline_v2.json ]; do sleep 40; done
sleep 20

echo "[t1 1/3] joint (lane type + colour), BASE model"
$P local/tlb/eval_joint_v1.py --question joint --out outputs/tlb/joint_base.json \
   > outputs/tlb/log_joint_base.log 2>&1
grep -E "colour only|lane type only|JOINT|gt joint|pred joint|sample" outputs/tlb/log_joint_base.log

echo "[t1 2/3] joint, FINE-TUNED model"
$P local/tlb/eval_joint_v1.py --question joint --lora outputs/tlb/lora_vqa_nb.pt \
   --out outputs/tlb/joint_ft.json > outputs/tlb/log_joint_ft.log 2>&1
grep -E "colour only|lane type only|JOINT|pred joint" outputs/tlb/log_joint_ft.log

echo "[t1 3/3] lane type, FINE-TUNED model (transfer check: LoRA saw only lights)"
$P local/tlb/eval_lanetype.py --question plain --lora outputs/tlb/lora_vqa_nb.pt \
   --out outputs/tlb/lanetype_ft.json > outputs/tlb/log_lanetype_ft.log 2>&1
grep -E "exact set|macro F1|^    left|^    straight|^    right" outputs/tlb/log_lanetype_ft.log
echo "TASK1_V2 QUEUE DONE"
