#!/usr/bin/env bash
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
# marker already present; run immediately
echo "=== choosing the pipeline configuration on the selector holdout (52 episodes) ==="
echo "    val is NOT read here"
for SEL in selector selector_v2; do
  for CFG in "argmax 1" "soft 3" "soft 6"; do
    set -- $CFG; V=$1; K=$2
    tag="${SEL}_${V}${K}"
    $P local/tlb/pipeline.py --selector outputs/tlb/$SEL.pt \
       --colour outputs/tlb/colour_head_v2.pt --vote $V --topk $K \
       --roi data/tlb/roi/train.npz --gt data/tlb/selcfg.jsonl \
       --det-gt data/tlb/det_train.jsonl \
       --out outputs/tlb/cfg_$tag.json > outputs/tlb/log_cfg_$tag.log 2>&1
    printf "  %-24s %s\n" "$tag" "$(grep -E '^  all ' outputs/tlb/log_cfg_$tag.log | sed 's/^  all *//')"
  done
done
echo "CFG SWEEP DONE"
