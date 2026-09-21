#!/usr/bin/env bash
# Train the gate, then score the decomposed VQA: gate says yes -> v1's 3-way colour answer
# (94.7% where a light exists), gate says no -> "none". Each half tuned for its own job.
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
CK=outputs/tlb/lora_gate_v1.pt
for a in 1 2 3 4 5 6; do
  E=""; [ -f $CK ] && E="--resume"
  $P local/tlb/train_gate_v1.py --train data/tlb/ft_gate_train.jsonl --cond full \
     --rank 16 --alpha 32 --epochs 1 --accum 8 --lr 1e-4 --save-every 25 $E --save $CK \
     >> outputs/tlb/log_gate.log 2>&1
  rc=$?; tail -1 outputs/tlb/log_gate.log
  [ $rc -eq 0 ] && break
  echo "[gate crashed rc=$rc, resuming]"; sleep 20
done
echo "[gate eval] over the whole validation split"
$P local/tlb/eval_gate_v1.py --lora $CK --out outputs/tlb/gate_val.json \
   > outputs/tlb/log_gate_eval.log 2>&1
grep -E "gate |decomposed|combined|for comparison|  v" outputs/tlb/log_gate_eval.log
echo "GATE DONE"
