#!/usr/bin/env bash
# The decomposition scored 93.9% only because the colour file covered exactly the 1300
# frames that HAVE a light -- every other frame fell through to "none" for free, which is
# the answer. Score the 3-way colour model over ALL 6019 frames first, then combine.
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
echo "[1/2] 3-way colour model over the WHOLE split (it has no 'none' option by design)"
$P local/tlb/eval_vqa.py --src data/tlb/full_val.jsonl --cond full --question assoc \
   --lora outputs/tlb/lora_vqa_nb.pt --out outputs/tlb/full_colour3_v1.json \
   > outputs/tlb/log_full_colour3.log 2>&1
grep -E "^  all " outputs/tlb/log_full_colour3.log | head -1
echo "[2/2] gate + colour, scored honestly"
$P local/tlb/eval_gate_v1.py --lora outputs/tlb/lora_gate_v1.pt \
   --colour-ft outputs/tlb/full_colour3_v1.json \
   --out outputs/tlb/gate_val_fixed.json > outputs/tlb/log_gate_eval_fixed.log 2>&1
grep -A 16 "===== GATE" outputs/tlb/log_gate_eval_fixed.log
echo "GATE_FIX DONE"
