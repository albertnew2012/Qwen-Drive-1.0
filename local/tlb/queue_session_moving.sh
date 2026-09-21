#!/usr/bin/env bash
# Session video on a MOVING scene. scene-0553 (the first one) sat at a red light at
# 0.0 km/h, so the planner's 5 s trajectory was a 0.4 m stub and told you nothing.
# scene-0796 covers 237 m at 43.5 km/h and passes a green light.
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
$P local/tlb/nuscenes_session_tl_v2.py --scene 5 --rate sweep \
   --out outputs/tlb/session_moving > outputs/tlb/log_session_moving.log 2>&1
tail -3 outputs/tlb/log_session_moving.log
N=$(ls outputs/tlb/session_moving/frames/*.png 2>/dev/null | wc -l)
FR=$($P -c "import json;print(json.load(open('outputs/tlb/session_moving/encode.json'))['framerate'])" 2>/dev/null || echo 7.5)
ffmpeg -y -loglevel error -framerate "$FR" -i outputs/tlb/session_moving/frames/%04d.png \
  -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:white,fps=30' -c:v libx264 -pix_fmt yuv420p -crf 20 \
  outputs/tlb/demo_v2/session_moving_all_perception.mp4
ls -la outputs/tlb/demo_v2/session_moving_all_perception.mp4 | awk '{printf "  %s  %.1f MB  (%s frames)\n",$9,$5/1e6,'"$N"'}'
echo "SESSION_MOVING DONE"
