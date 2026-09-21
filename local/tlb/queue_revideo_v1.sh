#!/usr/bin/env bash
# Re-dump the demo sessions with the consistency constraint and re-render, so the shipped
# videos match the fixed code rather than the version that drew contradictory ego sets.
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
S="outputs/tlb/selector_v2.pt outputs/tlb/selector_s1.pt outputs/tlb/selector_s2.pt outputs/tlb/selector_s3.pt outputs/tlb/selector_s4.pt"
$P local/tlb/pipeline_e2e_v4.py --det outputs/tlb/det_head_v2.pt --thr 0.30 --selector $S \
   --track 0.5 --ego-thr 0.35 --consistent --dump-boxes \
   --gt data/tlb/demo_segments.jsonl --out outputs/tlb/demo_dump_fixed.json \
   > outputs/tlb/log_demo_dump_fixed.log 2>&1
grep -E "^  all " outputs/tlb/log_demo_dump_fixed.log
$P - <<'PY'
import json
B='/home/albert/Desktop/Qwen-Drive-1.0/'
d=json.load(open(B+'outputs/tlb/demo_dump_fixed.json'))
inc=sum(x.get('ego_raw_inconsistent',0) for x in d)
mixed=sum(1 for x in d if len({x['box_colour'][i] for i in x.get('ego_set',[])
                               if i < len(x['box_colour'])})>1)
print(f"  demo frames: raw sets that were contradictory {inc}/{len(d)}; "
      f"after the fix {mixed}/{len(d)}")
PY
$P local/tlb/make_best_video_v1.py --dump outputs/tlb/demo_dump_fixed.json \
   --out outputs/tlb/demo_v2/best_perception_vs_vqa.mp4 > outputs/tlb/log_revideo.log 2>&1
tail -3 outputs/tlb/log_revideo.log
echo "REVIDEO DONE"
