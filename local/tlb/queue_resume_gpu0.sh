#!/usr/bin/env bash
# Resume after the power loss: GPU0 owns Task 3 (3D), which had not started training.
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
echo "[3D 1/3] train the 3D branch on the frozen 2D detector"
$P local/tlb/train_det3d_v2.py --epochs 3 --save outputs/tlb/det_head3d_v2.pt \
   > outputs/tlb/log_det3d_v2.log 2>&1
tail -4 outputs/tlb/log_det3d_v2.log | grep -vE "warn|Warn"
echo "[3D 2/3] Task 3 pipeline: 3D detection -> ego light"
$P local/tlb/pipeline_3d_v1.py --head3d outputs/tlb/det_head3d_v2.pt \
   --out outputs/tlb/pipeline3d_val.json > outputs/tlb/log_pipeline3d.log 2>&1
grep -E "colour, |range error|height error|predicted range" outputs/tlb/log_pipeline3d.log
echo "[3D 3/3] unfrozen variant, to see if joint training costs 2D accuracy"
$P local/tlb/train_det3d_v2.py --epochs 3 --unfreeze --lr 1e-4 \
   --save outputs/tlb/det_head3d_v2_unfrozen.pt > outputs/tlb/log_det3d_unf.log 2>&1
tail -3 outputs/tlb/log_det3d_unf.log | grep -vE "warn|Warn"
echo "RESUME_GPU0 DONE"
