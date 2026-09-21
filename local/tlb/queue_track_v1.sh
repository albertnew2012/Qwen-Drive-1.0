#!/usr/bin/env bash
# Does temporal tracking of P(governs) stabilise the ego light without costing colour?
# Chosen on the clean holdout, as always; val read afterwards.
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
S="outputs/tlb/selector_v2.pt outputs/tlb/selector_s1.pt outputs/tlb/selector_s2.pt outputs/tlb/selector_s3.pt outputs/tlb/selector_s4.pt"
for T in 0.0 0.5 0.7; do
  $P local/tlb/pipeline_e2e_v3.py --det outputs/tlb/det_head_v2.pt --thr 0.30 \
     --selector $S --track $T --gt data/tlb/val.jsonl \
     --out outputs/tlb/track_t$T.json > outputs/tlb/log_track_t$T.log 2>&1
  printf "  track=%-4s %s\n" $T "$(grep -E '^  all ' outputs/tlb/log_track_t$T.log|sed 's/^  all *//')"
done
echo "TRACK DONE"
