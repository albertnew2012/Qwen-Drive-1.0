#!/usr/bin/env bash
# GPU0: finish Task 2's finer detector, then train Task 3's 3D branch on frozen 2D.
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
while pgrep -f "[t]rain_det.py --epochs 4 --up 4" >/dev/null; do sleep 30; done

echo "[1/3] 4px-grid detector: detection metrics"
$P local/tlb/eval_det.py --ckpt outputs/tlb/det_head_up4.pt \
   --out outputs/tlb/det_eval_up4.json > outputs/tlb/log_det4_eval.log 2>&1
grep -E "IoU|colour accuracy" outputs/tlb/log_det4_eval.log

echo "[2/3] Task 3: train the 3D branch (2D frozen)"
$P local/tlb/train_det3d_v2.py --epochs 3 --save outputs/tlb/det_head3d_v2.pt \
   > outputs/tlb/log_det3d_v2.log 2>&1
tail -4 outputs/tlb/log_det3d_v2.log

echo "[3/3] Task 3: same, but whole head unfrozen, for comparison"
$P local/tlb/train_det3d_v2.py --epochs 3 --unfreeze --lr 1e-4 \
   --save outputs/tlb/det_head3d_v2_unfrozen.pt > outputs/tlb/log_det3d_v2_unfrozen.log 2>&1
tail -3 outputs/tlb/log_det3d_v2_unfrozen.log
echo "GPU0_V2 QUEUE DONE"
