#!/usr/bin/env bash
# Daytime turning scene. scene-0061 swings 98 degrees -- the largest turn available -- in
# clear daylight, with traffic lights annotated in 18 of 39 frames plus pedestrians and a
# crosswalk. The earlier turn video (scene-1094) was at night and hard to read.
# Note: this segment has no ego-governing colour LABEL, so the traffic-light overlay is
# shown as a live prediction with nothing to score it against.
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
$P local/tlb/nuscenes_session_tl_v2.py --scene 0 --rate sweep \
   --out outputs/tlb/session_day_turn > outputs/tlb/log_session_day_turn.log 2>&1
tail -2 outputs/tlb/log_session_day_turn.log
N=$(ls outputs/tlb/session_day_turn/frames/*.png 2>/dev/null | wc -l)
FR=$($P -c "import json;print(json.load(open('outputs/tlb/session_day_turn/encode.json'))['framerate'])" 2>/dev/null || echo 7.5)
ffmpeg -y -loglevel error -framerate "$FR" -i outputs/tlb/session_day_turn/frames/%04d.png \
  -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:white,fps=30' -c:v libx264 -pix_fmt yuv420p -crf 20 \
  outputs/tlb/demo_v2/session_day_turn_all_perception.mp4
ls -la outputs/tlb/demo_v2/session_day_turn_all_perception.mp4 | awk '{printf "  %s  %.1f MB  ('"$N"' frames)\n",$9,$5/1e6}'
echo "SESSION_DAY_TURN DONE"
