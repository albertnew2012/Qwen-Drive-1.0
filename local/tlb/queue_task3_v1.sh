#!/usr/bin/env bash
# Task 3: once the 3D branch is trained, score it two ways --
#   3D accuracy (range/height against the self-derived labels)
#   ego-light colour, the same metric Task 2 reports, so the paths are comparable
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
while [ ! -f outputs/tlb/det_head3d_v2.pt ]; do sleep 40; done
sleep 15
echo "[t3 1/2] 3D pipeline, frozen-2D variant"
$P local/tlb/pipeline_3d_v1.py --head3d outputs/tlb/det_head3d_v2.pt \
   --out outputs/tlb/pipeline3d_val.json > outputs/tlb/log_pipeline3d.log 2>&1
grep -E "colour, |range error|height error|predicted range" outputs/tlb/log_pipeline3d.log
if [ -f outputs/tlb/det_head3d_v2_unfrozen.pt ]; then
  echo "[t3 2/2] 3D pipeline, unfrozen variant (does joint training cost 2D accuracy?)"
  $P local/tlb/pipeline_3d_v1.py --head3d outputs/tlb/det_head3d_v2_unfrozen.pt \
     --out outputs/tlb/pipeline3d_val_unfrozen.json > outputs/tlb/log_pipeline3d_unf.log 2>&1
  grep -E "colour, |range error|height error" outputs/tlb/log_pipeline3d_unf.log
fi
echo "TASK3_V1 QUEUE DONE"
