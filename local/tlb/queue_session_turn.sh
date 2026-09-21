#!/usr/bin/env bash
# Session video on a TURNING scene. scene-1094 swings 74 degrees at 23 km/h and carries
# 10 green traffic-light frames, all of them discriminative -- so the planned trajectory
# curves, and the traffic-light association is being asked a hard question at the same time.
# Writes to its own filename; the straight-road and stationary videos are left alone.
set -u
cd /home/albert/Desktop/Qwen-Drive-1.0
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
$P local/tlb/nuscenes_session_tl_v2.py --scene 8 --rate sweep \
   --out outputs/tlb/session_turn > outputs/tlb/log_session_turn.log 2>&1
tail -2 outputs/tlb/log_session_turn.log
N=$(ls outputs/tlb/session_turn/frames/*.png 2>/dev/null | wc -l)
FR=$($P -c "import json;print(json.load(open('outputs/tlb/session_turn/encode.json'))['framerate'])" 2>/dev/null || echo 7.5)
ffmpeg -y -loglevel error -framerate "$FR" -i outputs/tlb/session_turn/frames/%04d.png \
  -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:white,fps=30' -c:v libx264 -pix_fmt yuv420p -crf 20 \
  outputs/tlb/demo_v2/session_turn_all_perception.mp4
ls -la outputs/tlb/demo_v2/session_turn_all_perception.mp4 | awk '{printf "  %s  %.1f MB  ('"$N"' frames)\n",$9,$5/1e6}'
echo "SESSION_TURN DONE"
