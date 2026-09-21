#!/usr/bin/env bash
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
# runs after the LoRA retrain frees GPU1
while [ ! -f outputs/tlb/ft_nb_assoc_full.json ]; do sleep 60; done
echo "[lane 1/2] base model, ego lane position"
$P local/tlb/eval_lane.py --question of --out outputs/tlb/lane_base.json \
   > outputs/tlb/log_lane_base.log 2>&1
grep -E "===== |^  all |^  multi|gt totals|predicted totals|unparseable|sample" outputs/tlb/log_lane_base.log
echo "[lane 2/2] lane count only"
$P local/tlb/eval_lane.py --question count --out outputs/tlb/lane_count.json \
   > outputs/tlb/log_lane_count.log 2>&1
grep -E "^  all |^  multi|predicted totals" outputs/tlb/log_lane_count.log
echo "LANE QUEUE DONE"
