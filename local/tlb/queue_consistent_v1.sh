#!/usr/bin/env bash
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
S="outputs/tlb/selector_v2.pt outputs/tlb/selector_s1.pt outputs/tlb/selector_s2.pt outputs/tlb/selector_s3.pt outputs/tlb/selector_s4.pt"
echo "[1/2] full val, ego set as-is -- how often is it self-contradictory?"
$P local/tlb/pipeline_e2e_v4.py --det outputs/tlb/det_head_v2.pt --thr 0.30 --selector $S \
   --track 0.5 --ego-thr 0.35 --dump-boxes --gt data/tlb/val.jsonl \
   --out outputs/tlb/egoset_raw.json > outputs/tlb/log_egoset_raw.log 2>&1
grep -E "^  all " outputs/tlb/log_egoset_raw.log
echo "[2/2] same, with the consistency constraint"
$P local/tlb/pipeline_e2e_v4.py --det outputs/tlb/det_head_v2.pt --thr 0.30 --selector $S \
   --track 0.5 --ego-thr 0.35 --consistent --dump-boxes --gt data/tlb/val.jsonl \
   --out outputs/tlb/egoset_consistent.json > outputs/tlb/log_egoset_cons.log 2>&1
grep -E "^  all " outputs/tlb/log_egoset_cons.log
echo "CONSISTENT DONE"
