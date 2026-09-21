#!/usr/bin/env bash
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
echo "=== choosing the detector threshold on the holdout (val NOT read) ==="
for T in 0.10 0.20 0.30 0.50; do
  $P local/tlb/pipeline_e2e.py --thr $T --gt data/tlb/selcfg.jsonl \
     --out outputs/tlb/e2e_cfg_t$T.json > outputs/tlb/log_e2e_cfg_t$T.log 2>&1
  printf "  thr=%-5s %s\n" $T "$(grep -E '^  all ' outputs/tlb/log_e2e_cfg_t$T.log | sed 's/^  all *//')"
done
echo "E2E THR CHOICE DONE"
