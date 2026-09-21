#!/usr/bin/env bash
# The ensemble on the FULL validation split, which is the headline denominator.
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
S="outputs/tlb/selector_v2.pt outputs/tlb/selector_s1.pt outputs/tlb/selector_s2.pt outputs/tlb/selector_s3.pt outputs/tlb/selector_s4.pt"
while pgrep -f "[t]rain_vqa_v2.py" >/dev/null; do sleep 60; done
echo "[full-ens] all 6019 frames, gov-thr 0.50 (holdout-chosen by the parsimony rule)"
$P local/tlb/pipeline_e2e_v2.py --det outputs/tlb/det_head_v2.pt --thr 0.30 --gov-thr 0.50 \
   --selector $S --gt data/tlb/full_val.jsonl \
   --out outputs/tlb/full_pipeline_ens.json > outputs/tlb/log_full_pipeline_ens.log 2>&1
grep -E "^  all |confusion" outputs/tlb/log_full_pipeline_ens.log
echo "FULL_ENS DONE"
