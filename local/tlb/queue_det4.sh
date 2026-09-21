#!/usr/bin/env bash
# Detector on a 4 px grid instead of 8. Recall stalled at ~80% when epochs were doubled,
# and half the end-to-end errors are detection misses on lamps around 16 px -- which span
# only two cells at 8 px. Finer cells are the targeted fix, not more training.
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
while pgrep -f "[p]ipeline_e2e.py" >/dev/null; do sleep 30; done
echo "[det4] probe"
$P local/tlb/train_det.py --probe --up 4 > outputs/tlb/log_det4_probe.log 2>&1
tail -4 outputs/tlb/log_det4_probe.log
grep -q "loss:" outputs/tlb/log_det4_probe.log || { echo "DET4 PROBE FAILED"; exit 1; }
echo "[det4] train 4 epochs at up=4"
$P local/tlb/train_det.py --epochs 4 --up 4 --save outputs/tlb/det_head_up4.pt \
   > outputs/tlb/log_det4_train.log 2>&1
tail -2 outputs/tlb/log_det4_train.log
echo "[det4] detection metrics"
$P local/tlb/eval_det.py --ckpt outputs/tlb/det_head_up4.pt --out outputs/tlb/det_eval_up4.json \
   > outputs/tlb/log_det4_eval.log 2>&1
grep -E "IoU|colour accuracy" outputs/tlb/log_det4_eval.log
echo "DET4 QUEUE DONE"
