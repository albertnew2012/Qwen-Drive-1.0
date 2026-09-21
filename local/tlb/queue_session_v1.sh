#!/usr/bin/env bash
# Full session render for scene-0553 (= OpenLane-V2 segment 11057, 41 red-light frames):
# every perception output, the planned trajectory, and the ego-lane traffic light together.
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
$P local/tlb/nuscenes_session_tl_v1.py --scene 2 --rate sweep --out outputs/tlb/session_tl \
   > outputs/tlb/log_session_tl.log 2>&1
tail -3 outputs/tlb/log_session_tl.log
N=$(ls outputs/tlb/session_tl/frames/*.png 2>/dev/null | wc -l)
echo "frames rendered: $N"
FR=$($P -c "import json;print(json.load(open('outputs/tlb/session_tl/encode.json'))['framerate'])" 2>/dev/null || echo 1.5)
ffmpeg -y -loglevel error -framerate "$FR" -i outputs/tlb/session_tl/frames/%04d.png \
  -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:white,fps=30' -c:v libx264 -pix_fmt yuv420p -crf 20 \
  outputs/tlb/demo_v2/session_all_perception.mp4
ls -la outputs/tlb/demo_v2/session_all_perception.mp4 | awk '{printf "  %s  %.1f MB\n",$9,$5/1e6}'
echo "SESSION DONE"
