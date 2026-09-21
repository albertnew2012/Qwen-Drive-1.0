#!/usr/bin/env bash
# Longer detector run. Recall at IoU 0.3 is 80.7% after 3 epochs and still improving;
# detection is the only remaining gap between the end-to-end system (96.5%) and the same
# pipeline on annotated boxes (98.4%).
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
while [ ! -f outputs/tlb/e2e_cfg_t0.50.json ]; do sleep 30; done
echo "[det2] 6 epochs"
$P local/tlb/train_det.py --epochs 6 --save outputs/tlb/det_head_v2.pt \
   > outputs/tlb/log_det_train2.log 2>&1
tail -2 outputs/tlb/log_det_train2.log
echo "[det2] detection metrics"
$P local/tlb/eval_det.py --ckpt outputs/tlb/det_head_v2.pt \
   --out outputs/tlb/det_eval_v2.json > outputs/tlb/log_det_eval_v2.log 2>&1
grep -E "IoU 0.3|IoU 0.5|colour accuracy" outputs/tlb/log_det_eval_v2.log
echo "[det2] end-to-end on the holdout, to pick the threshold"
for T in 0.10 0.30; do
  $P local/tlb/pipeline_e2e.py --det outputs/tlb/det_head_v2.pt --thr $T \
     --gt data/tlb/selcfg.jsonl --out outputs/tlb/e2e2_cfg_t$T.json \
     > outputs/tlb/log_e2e2_cfg_t$T.log 2>&1
  printf "  thr=%-5s %s\n" $T "$(grep -E '^  all ' outputs/tlb/log_e2e2_cfg_t$T.log|sed 's/^  all *//')"
done
echo "DET2 QUEUE DONE"
