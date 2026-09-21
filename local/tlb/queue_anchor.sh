#!/usr/bin/env bash
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
while pgrep -f "[e]val_vqa.py|[p]rompt_study.py" >/dev/null; do sleep 30; done
for Q in anchor14 anchor_none; do
  echo "[anchor] $Q"
  $P local/tlb/eval_lane.py --question $Q --stride 16 --out outputs/tlb/lane_$Q.json \
     > outputs/tlb/log_lane_$Q.log 2>&1
  grep -E "^  all |predicted totals" outputs/tlb/log_lane_$Q.log
done
echo "ANCHOR QUEUE DONE"
