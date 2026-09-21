#!/usr/bin/env bash
# Re-select the abstention threshold on segments the selector NEVER trained on.
# The first selection set shared 43% of its segments with selector training, so the
# selector was overconfident there and a high threshold looked safe; on val it
# over-abstained and cost 3.7 points.
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
for T in 0.20 0.30 0.40 0.50 0.60 0.70; do
  $P local/tlb/pipeline_e2e.py --det outputs/tlb/det_head_v2.pt --thr 0.30 --gov-thr $T \
     --gt data/tlb/full_holdout_clean.jsonl --out outputs/tlb/thrclean_t$T.json \
     > outputs/tlb/log_thrclean_t$T.log 2>&1
  printf "  gov-thr=%-5s %s\n" $T "$(grep -E '^  all ' outputs/tlb/log_thrclean_t$T.log|sed 's/^  all *//')"
done
echo "THR_CLEAN DONE"
