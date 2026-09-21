#!/usr/bin/env bash
# Demo videos, one per task. Uses the FROZEN 3D head: the control showed unfreezing buys
# 3D accuracy at a 5-point cost in colour, and colour is what the demo is about.
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
while pgrep -f "[p]ipeline_e2e_v2.py" >/dev/null; do sleep 30; done
echo "[demo 1/2] dump detections, selector scores, colours and 3D over the demo sessions"
$P local/tlb/dump_demo_v1.py --head3d outputs/tlb/det_head3d_v2.pt \
   --out outputs/tlb/demo_dump.json > outputs/tlb/log_demo_dump.log 2>&1
tail -2 outputs/tlb/log_demo_dump.log
echo "[demo 2/2] render three videos"
$P local/tlb/make_task_videos_v1.py --dump outputs/tlb/demo_dump.json \
   --vqa outputs/tlb/hint_ft.json > outputs/tlb/log_demo_render.log 2>&1
cat outputs/tlb/log_demo_render.log
ls -la outputs/tlb/demo/*.mp4 2>/dev/null | awk '{printf "  %-46s %6.1f MB\n",$9,$5/1e6}'
echo "DEMO_V2 DONE"
