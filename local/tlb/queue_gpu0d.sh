#!/usr/bin/env bash
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
while [ ! -f outputs/tlb/pipeline_v3.json ]; do sleep 20; done
echo "[det 1/3] probe -- shapes and one loss, before committing to a long run"
$P local/tlb/train_det.py --probe > outputs/tlb/log_det_probe.log 2>&1
tail -8 outputs/tlb/log_det_probe.log
if ! grep -q "loss:" outputs/tlb/log_det_probe.log; then
  echo "DET PROBE FAILED"; tail -20 outputs/tlb/log_det_probe.log; exit 1
fi
echo "[det 2/3] train the 2D traffic-light detector (frozen tower, dense head)"
$P local/tlb/train_det.py --epochs 3 --save outputs/tlb/det_head.pt \
   > outputs/tlb/log_det_train.log 2>&1
tail -3 outputs/tlb/log_det_train.log
echo "[det 3/3] detection metrics + downstream ego-light accuracy"
$P local/tlb/eval_det.py --ckpt outputs/tlb/det_head.pt > outputs/tlb/log_det_eval.log 2>&1
grep -A 20 "===== detection" outputs/tlb/log_det_eval.log | head -24
echo "GPU0D QUEUE DONE"
