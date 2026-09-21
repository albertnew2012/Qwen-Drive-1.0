#!/usr/bin/env bash
# Resume after the power loss: GPU1 finishes the corrected full-val pipeline (the run was
# 800/6019 in when the machine died) and then Task 1's explicit-form evaluations.
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
echo "[1/4] FULL val pipeline at the holdout-chosen gov-thr 0.80"
$P local/tlb/pipeline_e2e.py --det outputs/tlb/det_head_v2.pt --thr 0.30 --gov-thr 0.80 \
   --gt data/tlb/full_val.jsonl --out outputs/tlb/full_pipeline_v2.json \
   > outputs/tlb/log_full_pipeline_v2.log 2>&1
grep -E "^  all |^  light |confusion" outputs/tlb/log_full_pipeline_v2.log
echo "[2/4] joint (lane type + colour), BASE"
$P local/tlb/eval_joint_v1.py --question joint --out outputs/tlb/joint_base.json \
   > outputs/tlb/log_joint_base.log 2>&1
grep -E "colour only|lane type only|JOINT|sample" outputs/tlb/log_joint_base.log
echo "[3/4] joint, FINE-TUNED"
$P local/tlb/eval_joint_v1.py --question joint --lora outputs/tlb/lora_vqa_nb.pt \
   --out outputs/tlb/joint_ft.json > outputs/tlb/log_joint_ft.log 2>&1
grep -E "colour only|lane type only|JOINT" outputs/tlb/log_joint_ft.log
echo "[4/4] lane type, FINE-TUNED (transfer check: the LoRA only ever saw lights)"
$P local/tlb/eval_lanetype.py --question plain --lora outputs/tlb/lora_vqa_nb.pt \
   --out outputs/tlb/lanetype_ft.json > outputs/tlb/log_lanetype_ft.log 2>&1
grep -E "exact set|macro F1|^    left|^    straight|^    right" outputs/tlb/log_lanetype_ft.log
echo "RESUME_GPU1 DONE"
