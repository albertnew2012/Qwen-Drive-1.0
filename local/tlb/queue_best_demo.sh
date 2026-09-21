#!/usr/bin/env bash
# Demo with the BEST of each approach:
#   perception = end-to-end pipeline, ensemble selector, ego SET (97.85%)
#   VQA        = LoRA v1 + position-hint prompt (94.69%)
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
while pgrep -f "[e]val_vqa.py|[e]val_gate_v1.py" >/dev/null; do sleep 30; done
S="outputs/tlb/selector_v2.pt outputs/tlb/selector_s1.pt outputs/tlb/selector_s2.pt outputs/tlb/selector_s3.pt outputs/tlb/selector_s4.pt"
echo "[1/2] re-dump the demo sessions with the ensemble selector and the ego SET"
$P local/tlb/pipeline_e2e_v3.py --det outputs/tlb/det_head_v2.pt --thr 0.30 \
   --selector $S --track 0.5 --ego-thr 0.35 --dump-boxes \
   --gt data/tlb/demo_segments.jsonl --out outputs/tlb/demo_dump_best.json \
   > outputs/tlb/log_demo_dump_best.log 2>&1
grep -E "^  all " outputs/tlb/log_demo_dump_best.log
echo "[2/2] render the best-vs-best video"
$P local/tlb/make_best_video_v1.py > outputs/tlb/log_best_video.log 2>&1
cat outputs/tlb/log_best_video.log | tail -4
echo "BEST_DEMO DONE"
