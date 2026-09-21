#!/usr/bin/env bash
# Choose single-vs-ensemble on the CLEAN holdout (65 segments the selectors never saw),
# then read val once. Picking the best seed by its val score would be selection on val.
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
S="outputs/tlb/selector_v2.pt outputs/tlb/selector_s1.pt outputs/tlb/selector_s2.pt outputs/tlb/selector_s3.pt outputs/tlb/selector_s4.pt"
echo "[ens 1/2] on the clean holdout"
$P local/tlb/pipeline_e2e_v2.py --det outputs/tlb/det_head_v2.pt --thr 0.30 --gov-thr 0.50 \
   --selector $S --gt data/tlb/full_holdout_clean.jsonl \
   --out outputs/tlb/ens_holdout.json > outputs/tlb/log_ens_holdout.log 2>&1
printf "  ensemble(5): %s\n" "$(grep -E '^  all ' outputs/tlb/log_ens_holdout.log|sed 's/^  all *//')"
printf "  single     : %s\n" "$(grep -E '^  all ' outputs/tlb/log_thrclean_t0.50.log|sed 's/^  all *//')"
echo "[ens 2/2] whichever wins there, score the usable val subset"
$P local/tlb/pipeline_e2e_v2.py --det outputs/tlb/det_head_v2.pt --thr 0.30 \
   --selector $S --gt data/tlb/val.jsonl \
   --out outputs/tlb/ens_val.json > outputs/tlb/log_ens_val.log 2>&1
grep -E "^  all |^  discriminative|^  light |confusion" outputs/tlb/log_ens_val.log
echo "ENS DONE"
