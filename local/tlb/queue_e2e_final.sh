#!/usr/bin/env bash
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
while pgrep -f "[e]val_vqa.py" >/dev/null; do sleep 30; done
echo "=== FINAL end-to-end on val: detector v2, thr=0.30 (both chosen on the holdout) ==="
$P local/tlb/pipeline_e2e.py --det outputs/tlb/det_head_v2.pt --thr 0.30 \
   --out outputs/tlb/e2e_final.json > outputs/tlb/log_e2e_final.log 2>&1
grep -E "^  all |^  discriminative|^  light |confusion" outputs/tlb/log_e2e_final.log
echo "E2E FINAL DONE"
